use super::SlashCommandExecutor;
use std::path::PathBuf;

impl SlashCommandExecutor {
    pub(crate) fn parse_gateway_runtime_options(args: &[String]) -> (Option<u16>, Option<PathBuf>) {
        let mut port: Option<u16> = None;
        let mut dir: Option<PathBuf> = None;
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--port" => {
                    if let Some(next) = args.get(idx + 1) {
                        if let Ok(p) = next.parse::<u16>() {
                            port = Some(p);
                        }
                        idx += 1;
                    }
                }
                "--dir" => {
                    if let Some(next) = args.get(idx + 1) {
                        dir = Some(PathBuf::from(next));
                        idx += 1;
                    }
                }
                _ => {
                    if current.starts_with("--") {
                        // Ignore unknown flags here; specific parsers consume what they need.
                    } else if port.is_none()
                        && let Ok(p) = current.parse::<u16>()
                    {
                        port = Some(p);
                    } else if dir.is_none() {
                        dir = Some(PathBuf::from(current));
                    }
                }
            }
            idx += 1;
        }

        (port, dir)
    }

    pub(crate) fn parse_gateway_setup_options(
        args: &[String],
    ) -> (
        bool,
        Option<crate::gateway_runtime::ForwardProviderKind>,
        Option<u16>,
        Option<PathBuf>,
    ) {
        let mut forward = false;
        let mut provider: Option<crate::gateway_runtime::ForwardProviderKind> = None;
        let mut filtered: Vec<String> = Vec::new();
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--forward" => {
                    forward = true;
                }
                "--provider" => {
                    if let Some(next) = args.get(idx + 1) {
                        provider = crate::gateway_runtime::ForwardProviderKind::parse(next);
                        idx += 1;
                    }
                }
                _ => filtered.push(args[idx].clone()),
            }
            idx += 1;
        }

        let (port, dir) = Self::parse_gateway_runtime_options(&filtered);
        (forward, provider, port, dir)
    }

    pub(crate) fn parse_gateway_forward_options(
        args: &[String],
    ) -> (
        Option<crate::gateway_runtime::ForwardProviderKind>,
        Option<u16>,
        Option<PathBuf>,
    ) {
        let mut provider: Option<crate::gateway_runtime::ForwardProviderKind> = None;
        let mut filtered: Vec<String> = Vec::new();
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--provider" => {
                    if let Some(next) = args.get(idx + 1) {
                        provider = crate::gateway_runtime::ForwardProviderKind::parse(next);
                        idx += 1;
                    }
                }
                _ => filtered.push(args[idx].clone()),
            }
            idx += 1;
        }

        let (port, dir) = Self::parse_gateway_runtime_options(&filtered);
        (provider, port, dir)
    }

    pub(crate) fn parse_gateway_route_options(
        args: &[String],
    ) -> (Option<String>, Option<String>, Option<PathBuf>) {
        let mut route_key: Option<String> = None;
        let mut target_agent_id: Option<String> = None;
        let mut dir: Option<PathBuf> = None;
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--agent" => {
                    if let Some(next) = args.get(idx + 1) {
                        target_agent_id = Some(next.clone());
                        idx += 1;
                    }
                }
                "--dir" => {
                    if let Some(next) = args.get(idx + 1) {
                        dir = Some(PathBuf::from(next));
                        idx += 1;
                    }
                }
                _ => {
                    if route_key.is_none() {
                        route_key = Some(current.to_string());
                    }
                }
            }
            idx += 1;
        }

        (route_key, target_agent_id, dir)
    }

    pub(crate) fn parse_gateway_twilio_options(
        args: &[String],
    ) -> (Option<PathBuf>, Option<String>) {
        let mut dir: Option<PathBuf> = None;
        let mut base_url: Option<String> = None;
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--dir" => {
                    if let Some(next) = args.get(idx + 1) {
                        dir = Some(PathBuf::from(next));
                        idx += 1;
                    }
                }
                "--url" => {
                    if let Some(next) = args.get(idx + 1) {
                        base_url = Some(next.clone());
                        idx += 1;
                    }
                }
                _ => {}
            }
            idx += 1;
        }

        (dir, base_url)
    }

    pub(crate) fn parse_gateway_llm_options(
        args: &[String],
    ) -> (Option<String>, Option<String>, Option<PathBuf>) {
        let mut provider: Option<String> = None;
        let mut model: Option<String> = None;
        let mut dir: Option<PathBuf> = None;
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--provider" => {
                    if let Some(next) = args.get(idx + 1) {
                        provider = Some(next.clone());
                        idx += 1;
                    }
                }
                "--model" => {
                    if let Some(next) = args.get(idx + 1) {
                        model = Some(next.clone());
                        idx += 1;
                    }
                }
                "--dir" => {
                    if let Some(next) = args.get(idx + 1) {
                        dir = Some(PathBuf::from(next));
                        idx += 1;
                    }
                }
                _ => {}
            }
            idx += 1;
        }

        (provider, model, dir)
    }

    pub(crate) fn parse_gateway_logs_options(args: &[String]) -> (Option<PathBuf>, usize, bool) {
        let mut dir: Option<PathBuf> = None;
        let mut lines: usize = 120;
        let mut include_forwarding = true;
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--dir" => {
                    if let Some(next) = args.get(idx + 1) {
                        dir = Some(PathBuf::from(next));
                        idx += 1;
                    }
                }
                "--lines" => {
                    if let Some(next) = args.get(idx + 1) {
                        if let Ok(parsed) = next.parse::<usize>() {
                            lines = parsed.max(1);
                        }
                        idx += 1;
                    }
                }
                "--forward" | "--forwarding" => {
                    include_forwarding = true;
                }
                "--no-forward" | "--no-forwarding" => {
                    include_forwarding = false;
                }
                _ => {}
            }
            idx += 1;
        }

        (dir, lines, include_forwarding)
    }

    pub(crate) fn parse_gateway_doctor_options(args: &[String]) -> (Option<PathBuf>, bool) {
        let mut dir: Option<PathBuf> = None;
        let mut include_all = false;
        let mut idx = 0usize;

        while idx < args.len() {
            let current = args[idx].as_str();
            match current {
                "--dir" => {
                    if let Some(next) = args.get(idx + 1) {
                        dir = Some(PathBuf::from(next));
                        idx += 1;
                    }
                }
                "--all" | "all" => {
                    include_all = true;
                }
                _ => {}
            }
            idx += 1;
        }

        (dir, include_all)
    }

    pub(crate) fn parse_gateway_all_option(args: &[String]) -> bool {
        args.iter().any(|arg| arg == "--all" || arg == "all")
    }
}
