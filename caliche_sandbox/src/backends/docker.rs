//! Docker-based sandbox backend using bollard.

use super::child_output::CappedOutput;
use crate::config::{NetworkPolicy, SandboxConfig};
use crate::paths::{validate_read_path, validate_write_path};
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::{Context, Result};
use async_trait::async_trait;
use bollard::Docker;
use bollard::body_full;
use bollard::container::LogOutput;
use bollard::exec::{CreateExecOptions, StartExecOptions};
use bollard::models::{ContainerCreateBody, HostConfig};
use bollard::query_parameters::{
    CreateContainerOptions, DownloadFromContainerOptionsBuilder, RemoveContainerOptions,
    StartContainerOptions, UploadToContainerOptionsBuilder,
};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tar::{Builder as TarBuilder, Header};

const CONTAINER_WORKSPACE: &str = "/workspace";

#[derive(Debug)]
pub struct DockerBackend {
    docker: Docker,
    image: String,
}

impl DockerBackend {
    pub async fn new(image: &str) -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()?;
        docker.ping().await?;
        Ok(Self {
            docker,
            image: image.to_string(),
        })
    }
}

fn resolve_mount_host_path(path: &Path) -> Result<PathBuf> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            anyhow::bail!("refusing to mount symlink host path '{}'", path.display())
        }
        Ok(_) => path
            .canonicalize()
            .with_context(|| format!("canonicalize mount host path '{}'", path.display())),
        Err(err) => Err(err).with_context(|| format!("stat mount host path '{}'", path.display())),
    }
}

fn build_host_config(config: &SandboxConfig, workspace: &Path) -> Result<HostConfig> {
    let mut binds = vec![format!(
        "{}:{}:rw",
        workspace.display(),
        CONTAINER_WORKSPACE
    )];
    // The workspace `.git` (a dir, or the file a worktree or submodule
    // uses) is bound again read-only over the writable workspace, so
    // nothing in the container writes the config or hooks host git runs.
    // A symlinked `.git` is refused. A bind cannot keep a nested `.git` or
    // a submodule's `.git` read-only, because renaming a parent directory
    // moves the bind aside, so a workspace with one is refused. A bind
    // cannot cover a `.git` that does not exist, so a workspace without
    // one is refused too, as is a writable mount that holds the workspace
    // or its `.git`, and a git dir or common dir the container can write.
    refuse_unguarded_git(config, workspace)?;
    let host_git = resolve_mount_host_path(&workspace.join(crate::paths::GIT_DIR_NAME))?;
    binds.push(format!(
        "{}:{CONTAINER_WORKSPACE}/{}:ro",
        host_git.display(),
        crate::paths::GIT_DIR_NAME
    ));
    for mount in &config.mounts {
        let host_path = resolve_mount_host_path(&mount.host_path)?;
        let mode = if mount.read_only { "ro" } else { "rw" };
        binds.push(format!(
            "{}:{}:{mode}",
            host_path.display(),
            mount.container_path
        ));
    }

    let network_mode = match config.network {
        NetworkPolicy::Full => Some("bridge".to_string()),
        _ => Some("none".to_string()),
    };

    Ok(HostConfig {
        binds: Some(binds),
        network_mode,
        memory: Some((config.resource_limits.memory_mb.saturating_mul(1024 * 1024)) as i64),
        cpu_shares: Some(config.resource_limits.cpu_shares as i64),
        pids_limit: Some(config.resource_limits.pids_limit),
        ..Default::default()
    })
}

fn refuse_unguarded_git(config: &SandboxConfig, workspace: &Path) -> Result<()> {
    use crate::paths::git_refusal;
    let git = workspace.join(crate::paths::GIT_DIR_NAME);
    let mut writable = vec![workspace.to_path_buf()];
    for mount in config.mounts.iter().filter(|mount| !mount.read_only) {
        let host = resolve_mount_host_path(&mount.host_path)?;
        if workspace.starts_with(&host) {
            anyhow::bail!(git_refusal::docker_mount_contains_workspace(&host));
        }
        if host.starts_with(&git) {
            anyhow::bail!(git_refusal::overlapping_root(&host, &git));
        }
        writable.push(host);
    }
    let targets = crate::nested_git::guarded_targets(workspace, &writable)?;
    if !crate::git_targets::only_root(&targets) {
        let nested: Vec<String> = targets
            .iter()
            .filter(|path| *path != Path::new(crate::paths::GIT_DIR_NAME))
            .map(|path| path.display().to_string())
            .collect();
        anyhow::bail!(git_refusal::docker_nested(&nested.join(", ")));
    }
    if std::fs::symlink_metadata(&git).is_err() {
        anyhow::bail!(git_refusal::DOCKER_NO_ROOT_GIT);
    }
    crate::git_targets::refuse_exposed_git_dirs(workspace, &targets, &writable)?;
    crate::nested_git::refuse_git_outside(workspace, &writable)?;
    Ok(())
}

fn host_path_to_container_path(workspace: &Path, host_path: &Path) -> Result<String> {
    let workspace = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let host_path = host_path
        .canonicalize()
        .unwrap_or_else(|_| host_path.to_path_buf());
    if host_path == workspace {
        return Ok(CONTAINER_WORKSPACE.to_string());
    }
    if let Ok(relative) = host_path.strip_prefix(&workspace) {
        let joined = Path::new(CONTAINER_WORKSPACE).join(relative);
        return Ok(joined.to_string_lossy().replace('\\', "/"));
    }
    anyhow::bail!(
        "path '{}' is outside mounted workspace '{}'",
        host_path.display(),
        workspace.display()
    )
}

#[async_trait]
impl SandboxBackend for DockerBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: true,
            filesystem_isolation: true,
            tcp_isolation: true,
            resource_limits: true,
            host_mounts: true,
            pooling: true,
            backend_name: "docker",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        let workspace = workspace
            .canonicalize()
            .with_context(|| format!("canonicalize workspace '{}'", workspace.display()))?;
        let name = format!("caliche-sandbox-{}", uuid::Uuid::now_v7());
        let host_config = build_host_config(config, &workspace)?;
        let body = ContainerCreateBody {
            image: Some(self.image.clone()),
            cmd: Some(vec!["sleep".into(), "infinity".into()]),
            host_config: Some(host_config),
            working_dir: Some(CONTAINER_WORKSPACE.to_string()),
            ..Default::default()
        };
        let opts = CreateContainerOptions {
            name: Some(name),
            ..Default::default()
        };
        let create_response = self
            .docker
            .create_container(Some(opts), body)
            .await
            .map_err(|e| anyhow::anyhow!("Docker create_container failed: {e}"))?;
        let id = create_response.id;
        self.docker
            .start_container(&id, None::<StartContainerOptions>)
            .await
            .map_err(|e| anyhow::anyhow!("Docker start_container failed: {e}"))?;
        Ok(Box::new(DockerInstance {
            docker: self.docker.clone(),
            container_id: id,
            workspace,
            config: config.clone(),
        }))
    }
}

#[derive(Debug)]
struct DockerInstance {
    docker: Docker,
    container_id: String,
    workspace: PathBuf,
    config: SandboxConfig,
}

async fn drain_tar_bytes(
    stream: impl futures_util::Stream<Item = Result<bytes::Bytes, bollard::errors::Error>> + Unpin,
) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    let mut stream = stream;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| anyhow::anyhow!("Docker archive stream failed: {e}"))?;
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

#[async_trait]
impl SandboxInstance for DockerInstance {
    fn id(&self) -> &str {
        &self.container_id
    }

    async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult> {
        let start = Instant::now();
        let cmd: Vec<String> = command.iter().map(|s| (*s).to_string()).collect();
        let env_vec: Option<Vec<String>> =
            env.map(|m| m.iter().map(|(k, v)| format!("{k}={v}")).collect());
        let working_dir = match working_dir {
            Some(dir) => {
                let host = PathBuf::from(dir);
                Some(host_path_to_container_path(&self.workspace, &host)?)
            }
            None => Some(CONTAINER_WORKSPACE.to_string()),
        };
        let exec_opts = CreateExecOptions {
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            attach_stdin: Some(false),
            cmd: Some(cmd),
            working_dir,
            env: env_vec,
            ..Default::default()
        };
        let create_result = self
            .docker
            .create_exec(&self.container_id, exec_opts)
            .await
            .map_err(|e| anyhow::anyhow!("Docker create_exec failed: {e}"))?;
        let exec_id = create_result.id;
        let start_opts = StartExecOptions {
            output_capacity: Some(1024 * 1024),
            ..Default::default()
        };
        let output = self
            .docker
            .start_exec(&exec_id, Some(start_opts))
            .await
            .map_err(|e| anyhow::anyhow!("Docker start_exec failed: {e}"))?;
        let (mut stdout, mut stderr) = (CappedOutput::default(), CappedOutput::default());
        if let bollard::exec::StartExecResults::Attached {
            output: mut stream, ..
        } = output
        {
            let drain = async {
                while let Some(Ok(msg)) = stream.next().await {
                    match msg {
                        LogOutput::StdOut { message } => {
                            stdout.push(message.as_ref());
                        }
                        LogOutput::StdErr { message } => {
                            stderr.push(message.as_ref());
                        }
                        _ => {}
                    }
                }
            };
            if tokio::time::timeout(timeout, drain).await.is_err() {
                anyhow::bail!("Exec timed out after {:?}", timeout);
            }
        }
        let inspect = self
            .docker
            .inspect_exec(&exec_id)
            .await
            .map_err(|e| anyhow::anyhow!("Docker inspect_exec failed: {e}"))?;
        let exit_code = inspect.exit_code.unwrap_or(-1);
        let (stdout, stdout_truncated) = stdout.into_text();
        let (stderr, stderr_truncated) = stderr.into_text();
        Ok(ExecResult {
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
            exit_code,
            duration: start.elapsed(),
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let host_path = validate_read_path(&self.workspace, &self.config, path)?;
        let container_path = host_path_to_container_path(&self.workspace, &host_path)?;
        let download_path = container_path.replace('\\', "/");
        let options = DownloadFromContainerOptionsBuilder::default()
            .path(&download_path)
            .build();
        let stream = self
            .docker
            .download_from_container(&self.container_id, Some(options));
        let tar_bytes = drain_tar_bytes(stream).await?;
        let file_name = Path::new(&container_path)
            .file_name()
            .and_then(|name| name.to_str())
            .context("container path must include a file name")?;
        let mut archive = tar::Archive::new(tar_bytes.as_slice());
        for entry in archive.entries().context("read docker archive entries")? {
            let mut entry = entry.context("read docker archive entry")?;
            let entry_path = entry.path().context("archive entry path")?;
            if entry_path.file_name().and_then(|n| n.to_str()) == Some(file_name) {
                let mut content = Vec::new();
                entry
                    .read_to_end(&mut content)
                    .context("read archived file body")?;
                return Ok(content);
            }
        }
        anyhow::bail!("file '{}' not found in container archive", container_path)
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        let host_path = validate_write_path(&self.workspace, &self.config, path)?;
        let container_path = host_path_to_container_path(&self.workspace, &host_path)?;
        let container_parent = Path::new(&container_path)
            .parent()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| CONTAINER_WORKSPACE.to_string());
        let file_name = Path::new(&container_path)
            .file_name()
            .and_then(|name| name.to_str())
            .context("container path must include a file name")?;

        let mut tar_buf = Vec::new();
        {
            let mut builder = TarBuilder::new(&mut tar_buf);
            let mut header = Header::new_gnu();
            header.set_path(file_name).context("tar header path")?;
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append(&header, content)
                .context("append tar entry")?;
            builder.finish().context("finish tar archive")?;
        }

        let options = UploadToContainerOptionsBuilder::default()
            .path(&container_parent)
            .build();
        self.docker
            .upload_to_container(
                &self.container_id,
                Some(options),
                body_full(bytes::Bytes::from(tar_buf)),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Docker upload_to_container failed: {e}"))?;
        Ok(())
    }

    async fn is_alive(&self) -> bool {
        let inspect = self
            .docker
            .inspect_container(&self.container_id, None)
            .await;
        match inspect {
            Ok(info) => info.state.and_then(|s| s.running).unwrap_or(false),
            Err(_) => false,
        }
    }

    async fn destroy(&self) -> Result<()> {
        let opts = RemoveContainerOptions {
            force: true,
            ..Default::default()
        };
        self.docker
            .remove_container(&self.container_id, Some(opts))
            .await
            .map_err(|e| anyhow::anyhow!("Docker remove_container failed: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{build_host_config, resolve_mount_host_path};
    use crate::config::{MountConfig, NamedPolicy, SandboxBackendType, policy_to_config};
    use crate::paths::{GIT_DIR_NAME, GITFILE_PREFIX, git_refusal};
    use std::fs;
    use std::path::Path;

    fn binds_for(workspace: &Path) -> anyhow::Result<Vec<String>> {
        binds_with_mounts(workspace, Vec::new())
    }

    fn binds_with_mounts(
        workspace: &Path,
        mounts: Vec<MountConfig>,
    ) -> anyhow::Result<Vec<String>> {
        let mut config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            workspace,
            None,
            SandboxBackendType::Docker {
                image: "alpine".into(),
                pool_size: 0,
            },
        );
        config.mounts = mounts;
        Ok(build_host_config(&config, workspace)?
            .binds
            .unwrap_or_default())
    }

    #[test]
    fn the_workspace_git_dir_is_bound_read_only() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        crate::git_targets::tests::real_git_dir(&workspace.join(GIT_DIR_NAME));

        let binds = binds_for(&workspace).expect("binds");

        assert_eq!(
            binds,
            vec![
                format!("{}:/workspace:rw", workspace.display()),
                format!("{}/.git:/workspace/.git:ro", workspace.display()),
            ]
        );
    }

    /// A nested repository that no index lists is found by the scan, and
    /// the bind mount cannot keep it read-only.
    #[test]
    fn a_nested_repository_that_is_not_a_submodule_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        crate::git_targets::tests::real_git_dir(&workspace.join(GIT_DIR_NAME));
        crate::git_targets::tests::real_git_dir(&workspace.join("vendor").join(GIT_DIR_NAME));

        let binds = binds_for(&workspace).map_err(|err| err.to_string());

        assert_eq!(binds, Err(git_refusal::docker_nested("vendor/.git")));
    }

    /// The scan cannot rule out a `.git` in a directory nobody can list.
    #[cfg(unix)]
    #[test]
    fn a_directory_nobody_can_list_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        crate::git_targets::tests::real_git_dir(&workspace.join(GIT_DIR_NAME));
        let locked = workspace.join("locked");
        fs::create_dir(&locked).expect("locked");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).expect("chmod 000");
        fs::read_dir(&locked).expect_err("mode 000 is unlistable");

        let binds = binds_for(&workspace).map_err(|err| err.to_string());

        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).expect("chmod back");
        assert_eq!(binds, Err(git_refusal::unlistable_dir(&locked)));
    }

    #[test]
    fn a_workspace_with_nested_git_metadata_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        crate::git_targets::tests::real_git_dir(&workspace.join(GIT_DIR_NAME));
        fs::create_dir_all(workspace.join("sub")).expect("sub");
        fs::write(
            workspace.join("sub").join(GIT_DIR_NAME),
            format!(
                "{GITFILE_PREFIX} ../{GIT_DIR_NAME}/{}/sub\n",
                crate::paths::MODULES_DIR
            ),
        )
        .expect("gitfile");
        fs::write(
            workspace.join(GIT_DIR_NAME).join(crate::paths::INDEX_FILE),
            crate::git_targets::tests::gitlink_index("sub"),
        )
        .expect("index");

        let err = binds_for(&workspace).expect_err("submodule .git");

        assert_eq!(err.to_string(), git_refusal::docker_nested("sub/.git"));
    }

    fn mount(host_path: &Path, read_only: bool) -> MountConfig {
        MountConfig {
            host_path: host_path.to_path_buf(),
            container_path: "/data".into(),
            read_only,
        }
    }

    /// `git init --separate-git-dir <ws>/meta <ws>`.
    #[test]
    fn a_git_dir_inside_the_workspace_outside_git_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        fs::create_dir(workspace.join("meta")).expect("meta");
        crate::git_targets::tests::real_git_dir(&workspace.join("meta"));
        fs::write(
            workspace.join(GIT_DIR_NAME),
            format!("{GITFILE_PREFIX} meta\n"),
        )
        .expect("gitfile");

        let err = binds_for(&workspace).expect_err("refused");

        assert_eq!(
            err.to_string(),
            git_refusal::git_dir_in_workspace(&workspace.join("meta"))
        );
    }

    /// A writable mount around the workspace, or around the git dir its
    /// gitfile names, is refused; the same mounts read-only are not.
    #[test]
    fn a_writable_mount_that_reaches_git_metadata_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let base = dir.path().canonicalize().expect("canonical");
        let workspace = base.join("ws");
        let main = base.join("main");
        fs::create_dir_all(&workspace).expect("ws");
        let git_dir = caliche_policy::plant_linked_worktree(&main.join(GIT_DIR_NAME), "ws")
            .expect("linked worktree");
        fs::write(
            workspace.join(GIT_DIR_NAME),
            format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
        )
        .expect("gitfile");

        let around = binds_with_mounts(&workspace, vec![mount(&base, false)]).expect_err("around");
        let main_repo = binds_with_mounts(&workspace, vec![mount(&main, false)]).expect_err("main");
        let read_only = binds_with_mounts(&workspace, vec![mount(&base, true), mount(&main, true)]);

        assert_eq!(
            around.to_string(),
            git_refusal::docker_mount_contains_workspace(&base)
        );
        assert_eq!(
            main_repo.to_string(),
            git_refusal::git_dir_under_writable_root(&git_dir, &main)
        );
        assert!(read_only.is_ok(), "{read_only:?}");
    }

    /// A writable mount inside the workspace `.git` is refused; the same
    /// mount read-only is not.
    #[test]
    fn a_writable_mount_inside_the_workspace_git_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        let git = workspace.join(GIT_DIR_NAME);
        let hooks = git.join("hooks");
        crate::git_targets::tests::real_git_dir(&git);
        fs::create_dir_all(&hooks).expect("hooks");

        let writable = binds_with_mounts(&workspace, vec![mount(&hooks, false)]).expect_err("rw");
        let read_only = binds_with_mounts(&workspace, vec![mount(&hooks, true)]);

        assert_eq!(
            writable.to_string(),
            git_refusal::overlapping_root(&hooks, &git)
        );
        assert!(read_only.is_ok(), "{read_only:?}");
    }

    /// A writable mount inside a linked worktree's common dir (here the
    /// main repository's hooks) is refused.
    #[test]
    fn a_writable_mount_inside_the_common_dir_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let base = dir.path().canonicalize().expect("canonical");
        let workspace = base.join("ws");
        let common = base.join("main").join(GIT_DIR_NAME);
        let hooks = common.join("hooks");
        fs::create_dir_all(&workspace).expect("ws");
        let git_dir =
            caliche_policy::plant_linked_worktree(&common, "ws").expect("linked worktree");
        fs::create_dir_all(&hooks).expect("hooks");
        fs::write(
            workspace.join(GIT_DIR_NAME),
            format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
        )
        .expect("gitfile");

        let err = binds_with_mounts(&workspace, vec![mount(&hooks, false)]).expect_err("refused");

        assert_eq!(
            err.to_string(),
            git_refusal::overlapping_root(&hooks, &common)
        );
    }

    #[test]
    fn a_workspace_without_git_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");

        let err = binds_for(&workspace).expect_err("no root .git");

        assert_eq!(err.to_string(), git_refusal::DOCKER_NO_ROOT_GIT);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_workspace_git_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        fs::create_dir(workspace.join("real-git")).expect("real-git");
        std::os::unix::fs::symlink("real-git", workspace.join(GIT_DIR_NAME)).expect("symlink");

        let err = binds_for(&workspace).expect_err("symlinked .git");

        assert_eq!(
            err.to_string(),
            git_refusal::invalid_git_entry(&workspace.join(GIT_DIR_NAME))
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_objects_symlink_into_the_workspace_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let workspace = dir.path().canonicalize().expect("canonical");
        let git = workspace.join(GIT_DIR_NAME);
        crate::git_targets::tests::real_git_dir(&git);
        fs::create_dir(workspace.join("store")).expect("store");
        let objects = git.join(crate::paths::OBJECTS_DIR);
        fs::remove_dir(&objects).expect("objects");
        std::os::unix::fs::symlink("../store", &objects).expect("link");

        let err = binds_for(&workspace).expect_err("refused");

        assert_eq!(err.to_string(), git_refusal::git_dir_in_workspace(&objects));
    }

    #[cfg(unix)]
    #[test]
    fn resolve_mount_host_path_refuses_symlink() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let target = dir.path().join("target");
        fs::create_dir(&target).expect("target dir");
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let err = resolve_mount_host_path(&link).expect_err("symlink mount");
        assert!(
            err.to_string().contains("symlink"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_mount_host_path_accepts_real_directory() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let resolved = resolve_mount_host_path(dir.path()).expect("real dir");
        assert_eq!(resolved, dir.path().canonicalize().expect("canonical"));
    }
}
