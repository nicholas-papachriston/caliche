//! Plugins tab - guided editor for schema-based plugins.

use dioxus::prelude::*;
use serde_json::{Value, json};

use crate::api::{
    ApiClient, PluginCatalogResponse, PluginExportResponse, PluginGuide, PluginSummary,
    PluginValidationResponse,
};

#[derive(Debug, Clone, PartialEq)]
struct PluginEditorDraft {
    name: String,
    version: String,
    description: String,
    author: String,
    tags_csv: String,
    plugin_type: String,
    spec_json: String,
}

impl Default for PluginEditorDraft {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.1.0".to_string(),
            description: String::new(),
            author: String::new(),
            tags_csv: String::new(),
            plugin_type: "tool".to_string(),
            spec_json: "{}".to_string(),
        }
    }
}

impl PluginEditorDraft {
    fn with_guide(guide: &PluginGuide) -> Self {
        let mut draft = Self {
            plugin_type: guide.plugin_type.clone(),
            ..Self::default()
        };
        draft.apply_definition(&guide.starter_definition);
        draft
    }

    fn apply_definition(&mut self, value: &Value) {
        self.name = value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.version = value
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or("0.1.0")
            .to_string();
        self.description = value
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.author = value
            .get("author")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        self.tags_csv = value
            .get("tags")
            .and_then(Value::as_array)
            .map(|tags| {
                tags.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        self.plugin_type = value
            .get("plugin_type")
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_string();
        self.spec_json = value
            .get("spec")
            .map(pretty_json)
            .unwrap_or_else(|| "{}".to_string());
    }

    fn to_definition(&self) -> Result<Value, String> {
        let spec: Value = serde_json::from_str(&self.spec_json)
            .map_err(|err| format!("Spec JSON is invalid: {}", err))?;
        let tags = self
            .tags_csv
            .split(',')
            .map(str::trim)
            .filter(|tag| !tag.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let mut obj = json!({
            "name": self.name.trim(),
            "version": self.version.trim(),
            "description": self.description.trim(),
            "tags": tags,
            "plugin_type": self.plugin_type,
            "spec": spec,
        });
        if !self.author.trim().is_empty() {
            obj["author"] = json!(self.author.trim());
        }
        Ok(obj)
    }
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string())
}

fn field_row(label: &str, help: &str, control: Element) -> Element {
    rsx! {
        div { style: "display: flex; flex-direction: column; gap: 0.35em; margin-bottom: 1em;",
            div { style: "display: flex; justify-content: space-between; gap: 1ch; align-items: baseline;",
                span { style: "color: var(--text-color);", "{label}" }
                if !help.is_empty() {
                    span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "{help}" }
                }
            }
            {control}
        }
    }
}

fn selected_guide<'a>(
    catalog: &'a PluginCatalogResponse,
    plugin_type: &str,
) -> Option<&'a PluginGuide> {
    catalog
        .guides
        .iter()
        .find(|guide| guide.plugin_type == plugin_type)
}

fn apply_guide_selection(
    guides: &[PluginGuide],
    plugin_type: &str,
    mut draft: Signal<PluginEditorDraft>,
    mut validation: Signal<Option<PluginValidationResponse>>,
    mut export_data: Signal<Option<PluginExportResponse>>,
    mut status: Signal<Option<(bool, String)>>,
) {
    if let Some(guide) = guides.iter().find(|guide| guide.plugin_type == plugin_type) {
        draft.set(PluginEditorDraft::with_guide(guide));
        validation.set(None);
        export_data.set(None);
        status.set(Some((
            true,
            format!("Loaded {} starter template.", guide.title),
        )));
    }
}

fn load_plugin_into_editor(
    api: ApiClient,
    name: String,
    mut draft: Signal<PluginEditorDraft>,
    mut validation: Signal<Option<PluginValidationResponse>>,
    mut export_data: Signal<Option<PluginExportResponse>>,
    mut status: Signal<Option<(bool, String)>>,
    mut busy: Signal<bool>,
) {
    busy.set(true);
    spawn(async move {
        match api.get_plugin(&name).await {
            Ok(definition) => {
                let mut next = draft.read().clone();
                next.apply_definition(&definition);
                draft.set(next);
                validation.set(None);
                export_data.set(None);
                status.set(Some((true, format!("Loaded plugin `{}`.", name))));
            }
            Err(err) => {
                status.set(Some((
                    false,
                    format!("Failed to load plugin `{}`: {}", name, err),
                )));
            }
        }
        busy.set(false);
    });
}

/// Props for the plugins view.
#[derive(Props, Clone, PartialEq)]
pub struct PluginsViewProps {
    pub api_client: ApiClient,
}

#[component]
pub fn PluginsView(props: PluginsViewProps) -> Element {
    let mut catalog = use_signal(|| PluginCatalogResponse { guides: Vec::new() });
    let mut plugins = use_signal(Vec::<PluginSummary>::new);
    let mut draft = use_signal(PluginEditorDraft::default);
    let mut validation = use_signal(|| None::<PluginValidationResponse>);
    let mut status = use_signal(|| None::<(bool, String)>);
    let mut export_data = use_signal(|| None::<PluginExportResponse>);
    let mut import_json = use_signal(String::new);
    let mut loaded = use_signal(|| false);
    let mut busy = use_signal(|| false);

    let api = props.api_client.clone();
    use_effect(move || {
        if *loaded.read() {
            return;
        }
        loaded.set(true);
        let api = api.clone();
        spawn(async move {
            let mut next_status = None;
            match api.get_plugin_catalog().await {
                Ok(response) => {
                    let first_guide = response.guides.first().cloned();
                    catalog.set(response);
                    if let Some(guide) = first_guide {
                        draft.set(PluginEditorDraft::with_guide(&guide));
                    }
                }
                Err(err) => {
                    next_status = Some((false, format!("Failed to load plugin catalog: {}", err)));
                }
            }
            match api.get_plugins().await {
                Ok(response) => plugins.set(response.plugins),
                Err(err) => {
                    next_status = Some((false, format!("Failed to load plugins: {}", err)));
                }
            }
            status.set(next_status);
        });
    });

    let catalog_value = catalog.read().clone();
    let plugins_value = plugins.read().clone();
    let draft_value = draft.read().clone();
    let status_value = status.read().clone();
    let validation_value = validation.read().clone();
    let export_value = export_data.read().clone();
    let import_text = import_json.read().clone();
    let current_guide = selected_guide(&catalog_value, &draft_value.plugin_type).cloned();
    let preview_definition = draft_value.to_definition();
    let preview_text = preview_definition
        .as_ref()
        .map(pretty_json)
        .unwrap_or_else(|err| format!("Unable to build definition:\n{}", err));

    let validate_current = {
        let api = props.api_client.clone();
        move |_| {
            let api = api.clone();
            let definition = match draft.read().to_definition() {
                Ok(definition) => definition,
                Err(err) => {
                    validation.set(None);
                    status.set(Some((false, err)));
                    return;
                }
            };
            busy.set(true);
            spawn(async move {
                match api.validate_plugin(definition).await {
                    Ok(response) => {
                        validation.set(Some(response.clone()));
                        if response.valid {
                            status.set(Some((true, "Plugin definition validated.".to_string())));
                        } else {
                            status.set(Some((false, response.errors.join("\n"))));
                        }
                    }
                    Err(err) => {
                        validation.set(None);
                        status.set(Some((false, format!("Validation request failed: {}", err))));
                    }
                }
                busy.set(false);
            });
        }
    };

    let save_current = {
        let api = props.api_client.clone();
        move |_| {
            let api = api.clone();
            let definition = match draft.read().to_definition() {
                Ok(definition) => definition,
                Err(err) => {
                    status.set(Some((false, err)));
                    return;
                }
            };
            busy.set(true);
            spawn(async move {
                match api.save_plugin(definition).await {
                    Ok(response) => {
                        let mut next = draft.read().clone();
                        next.apply_definition(&response.definition);
                        draft.set(next);
                        validation.set(None);
                        export_data.set(None);
                        status.set(Some((
                            true,
                            if response.created {
                                format!("Created plugin `{}`.", response.summary.name)
                            } else {
                                format!("Updated plugin `{}`.", response.summary.name)
                            },
                        )));
                        if let Ok(list_response) = api.get_plugins().await {
                            plugins.set(list_response.plugins);
                        }
                    }
                    Err(err) => {
                        status.set(Some((false, format!("Failed to save plugin: {}", err))));
                    }
                }
                busy.set(false);
            });
        }
    };

    let export_current = {
        let api = props.api_client.clone();
        move |_| {
            let api = api.clone();
            let name = draft.read().name.trim().to_string();
            if name.is_empty() {
                status.set(Some((
                    false,
                    "Set a plugin name before exporting.".to_string(),
                )));
                return;
            }
            busy.set(true);
            spawn(async move {
                match api.export_plugin(&name).await {
                    Ok(response) => {
                        export_data.set(Some(response));
                        status.set(Some((true, format!("Exported plugin `{}`.", name))));
                    }
                    Err(err) => {
                        status.set(Some((
                            false,
                            format!("Failed to export plugin `{}`: {}", name, err),
                        )));
                    }
                }
                busy.set(false);
            });
        }
    };

    let delete_current = {
        let api = props.api_client.clone();
        let guides = catalog_value.guides.clone();
        move |_| {
            let api = api.clone();
            let guides = guides.clone();
            let name = draft.read().name.trim().to_string();
            let plugin_type = draft.read().plugin_type.clone();
            if name.is_empty() {
                status.set(Some((
                    false,
                    "Load or name a plugin before deleting it.".to_string(),
                )));
                return;
            }
            busy.set(true);
            spawn(async move {
                match api.delete_plugin(&name).await {
                    Ok(()) => {
                        status.set(Some((true, format!("Deleted plugin `{}`.", name))));
                        export_data.set(None);
                        validation.set(None);
                        if let Some(guide) =
                            guides.iter().find(|guide| guide.plugin_type == plugin_type)
                        {
                            draft.set(PluginEditorDraft::with_guide(guide));
                        } else {
                            draft.set(PluginEditorDraft::default());
                        }
                        if let Ok(list_response) = api.get_plugins().await {
                            plugins.set(list_response.plugins);
                        }
                    }
                    Err(err) => {
                        status.set(Some((
                            false,
                            format!("Failed to delete plugin `{}`: {}", name, err),
                        )));
                    }
                }
                busy.set(false);
            });
        }
    };

    let import_current = {
        move |_| {
            let text = import_json.read().clone();
            if text.trim().is_empty() {
                status.set(Some((
                    false,
                    "Paste exported plugin JSON first.".to_string(),
                )));
                return;
            }
            match serde_json::from_str::<Value>(&text) {
                Ok(value) => {
                    let mut next = draft.read().clone();
                    next.apply_definition(&value);
                    draft.set(next);
                    validation.set(None);
                    export_data.set(None);
                    status.set(Some((
                        true,
                        "Loaded plugin JSON into the editor.".to_string(),
                    )));
                }
                Err(err) => {
                    status.set(Some((false, format!("Import JSON is invalid: {}", err))));
                }
            }
        }
    };

    let input_style = "width: 100%; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5em 1ch;";
    let panel_style = "border: 1px solid var(--border-color); background: var(--card-bg); padding: 1em; display: flex; flex-direction: column; gap: 0.75em;";

    rsx! {
        div { class: "plugins-view",
            style: "flex: 1; min-height: 0; overflow: hidden; padding: 1ch; display: flex; gap: 1ch;",

            // Library / guide rail
            div { style: "width: 24rem; min-width: 20rem; max-width: 28rem; display: flex; flex-direction: column; gap: 1ch; overflow: hidden;",
                div { style: "{panel_style} overflow-y: auto;",
                    div { style: "display: flex; justify-content: space-between; align-items: center;",
                        span { style: "color: var(--accent);", "plugin types" }
                        button {
                            style: "background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); padding: 0.25em 0.75ch;",
                            onclick: move |_| {
                                if let Some(guide) = catalog.read().guides.first().cloned() {
                                    draft.set(PluginEditorDraft::with_guide(&guide));
                                    validation.set(None);
                                    export_data.set(None);
                                    status.set(Some((true, "Started a new plugin from the first guide.".to_string())));
                                }
                            },
                            "new"
                        }
                    }
                    for guide in &catalog_value.guides {
                        {
                            let guide_clone = guide.clone();
                            let selected = guide.plugin_type == draft_value.plugin_type;
                            rsx! {
                                button {
                                    key: "{guide_clone.plugin_type}",
                                    style: if selected {
                                        "text-align: left; background: var(--accent-dim); color: #000; border: 1px solid var(--accent); padding: 0.75em 1ch;"
                                    } else {
                                        "text-align: left; background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.75em 1ch;"
                                    },
                                    onclick: {
                                        let guides = catalog_value.guides.clone();
                                        let plugin_type = guide_clone.plugin_type.clone();
                                        move |_| apply_guide_selection(
                                            &guides,
                                            &plugin_type,
                                            draft,
                                            validation,
                                            export_data,
                                            status,
                                        )
                                    },
                                    strong { "{guide_clone.title}" }
                                    div { style: "font-size: var(--font-size-xs); margin-top: 0.35em; color: inherit;", "{guide_clone.summary}" }
                                }
                            }
                        }
                    }
                }

                div { style: "{panel_style} overflow-y: auto;",
                    span { style: "color: var(--accent);", "saved plugins" }
                    if plugins_value.is_empty() {
                        span { style: "color: var(--muted-color);", "No saved plugins yet." }
                    } else {
                        for plugin in &plugins_value {
                            {
                                let plugin_clone = plugin.clone();
                                rsx! {
                                    button {
                                        key: "{plugin_clone.name}",
                                        style: "text-align: left; background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.75em 1ch;",
                                        onclick: {
                                            let api = props.api_client.clone();
                                            move |_| load_plugin_into_editor(
                                                api.clone(),
                                                plugin_clone.name.clone(),
                                                draft,
                                                validation,
                                                export_data,
                                                status,
                                                busy,
                                            )
                                        },
                                        strong { "{plugin_clone.name}" }
                                        div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-top: 0.35em;",
                                            "{plugin_clone.plugin_type}  |  v{plugin_clone.version}"
                                        }
                                        div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-top: 0.25em;",
                                            "{plugin_clone.description}"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Editor
            div { style: "flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1ch; overflow: hidden;",
                if let Some((ok, message)) = status_value {
                    div {
                        style: if ok {
                            "padding: 0.75em 1ch; border: 1px solid var(--accent); background: var(--secondary-bg);"
                        } else {
                            "padding: 0.75em 1ch; border: 1px solid var(--error); background: var(--secondary-bg); color: var(--error);"
                        },
                        "{message}"
                    }
                }

                div { style: "display: flex; gap: 1ch; min-height: 0; overflow: hidden;",
                    div { style: "flex: 1.1; min-width: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 1ch;",
                        div { style: "{panel_style}",
                            span { style: "color: var(--accent);", "plugin envelope" }
                            {field_row("Name", "Machine-readable identifier.", rsx! {
                                input {
                                    style: "{input_style}",
                                    value: "{draft_value.name}",
                                    oninput: move |evt| {
                                        let mut next = draft.read().clone();
                                        next.name = evt.value();
                                        draft.set(next);
                                    }
                                }
                            })}
                            {field_row("Version", "Semantic version string.", rsx! {
                                input {
                                    style: "{input_style}",
                                    value: "{draft_value.version}",
                                    oninput: move |evt| {
                                        let mut next = draft.read().clone();
                                        next.version = evt.value();
                                        draft.set(next);
                                    }
                                }
                            })}
                            {field_row("Description", "Short summary shown in listings.", rsx! {
                                textarea {
                                    style: "{input_style} min-height: 5em;",
                                    value: "{draft_value.description}",
                                    oninput: move |evt| {
                                        let mut next = draft.read().clone();
                                        next.description = evt.value();
                                        draft.set(next);
                                    }
                                }
                            })}
                            {field_row("Author", "Optional owner or team.", rsx! {
                                input {
                                    style: "{input_style}",
                                    value: "{draft_value.author}",
                                    oninput: move |evt| {
                                        let mut next = draft.read().clone();
                                        next.author = evt.value();
                                        draft.set(next);
                                    }
                                }
                            })}
                            {field_row("Tags", "Comma-separated labels.", rsx! {
                                input {
                                    style: "{input_style}",
                                    value: "{draft_value.tags_csv}",
                                    oninput: move |evt| {
                                        let mut next = draft.read().clone();
                                        next.tags_csv = evt.value();
                                        draft.set(next);
                                    }
                                }
                            })}
                            {field_row("Plugin Type", "Select the runtime contract this plugin implements.", rsx! {
                                select {
                                    style: "{input_style}",
                                    value: "{draft_value.plugin_type}",
                                    onchange: {
                                        let guides = catalog_value.guides.clone();
                                        move |evt| apply_guide_selection(
                                            &guides,
                                            &evt.value(),
                                            draft,
                                            validation,
                                            export_data,
                                            status,
                                        )
                                    },
                                    for guide in &catalog_value.guides {
                                        option {
                                            value: "{guide.plugin_type}",
                                            "{guide.title}"
                                        }
                                    }
                                }
                            })}
                        }

                        if let Some(guide) = current_guide.clone() {
                            div { style: "{panel_style}",
                                span { style: "color: var(--accent);", "guided starter" }
                                div { "{guide.summary}" }
                                div { style: "color: var(--muted-color); font-size: var(--font-size-sm);",
                                    "{guide.when_to_use}"
                                }
                                for note in &guide.notes {
                                    div { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                                        "- {note}"
                                    }
                                }
                                button {
                                    style: "align-self: flex-start; background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0.35em 1ch;",
                                    onclick: {
                                        let guides = catalog_value.guides.clone();
                                        let plugin_type = guide.plugin_type.clone();
                                        move |_| apply_guide_selection(
                                            &guides,
                                            &plugin_type,
                                            draft,
                                            validation,
                                            export_data,
                                            status,
                                        )
                                    },
                                    "reset to starter"
                                }
                            }
                        }

                        div { style: "{panel_style}",
                            span { style: "color: var(--accent);", "type-specific spec" }
                            div { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                                "Edit only the `spec` object here. The outer plugin envelope is managed above."
                            }
                            textarea {
                                style: "{input_style} min-height: 24em; font-family: var(--font-mono);",
                                value: "{draft_value.spec_json}",
                                oninput: move |evt| {
                                    let mut next = draft.read().clone();
                                    next.spec_json = evt.value();
                                    draft.set(next);
                                }
                            }
                        }

                        div { style: "{panel_style}",
                            span { style: "color: var(--accent);", "import / export" }
                            textarea {
                                style: "{input_style} min-height: 10em; font-family: var(--font-mono);",
                                placeholder: "Paste exported plugin JSON here to load it into the editor.",
                                value: "{import_text}",
                                oninput: move |evt| import_json.set(evt.value()),
                            }
                            div { style: "display: flex; gap: 1ch; flex-wrap: wrap;",
                                button {
                                    style: "background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0.35em 1ch;",
                                    onclick: import_current,
                                    "load json"
                                }
                                button {
                                    style: "background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0.35em 1ch;",
                                    onclick: export_current,
                                    "export current"
                                }
                            }
                            if let Some(exported) = export_value {
                                textarea {
                                    style: "{input_style} min-height: 12em; font-family: var(--font-mono);",
                                    readonly: true,
                                    value: "{exported.definition}",
                                }
                            }
                        }
                    }

                    // Preview + actions
                    div { style: "flex: 0.9; min-width: 22rem; max-width: 42rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1ch;",
                        div { style: "{panel_style}",
                            span { style: "color: var(--accent);", "actions" }
                            div { style: "display: flex; gap: 1ch; flex-wrap: wrap;",
                                button {
                                    disabled: *busy.read(),
                                    style: "background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0.4em 1ch;",
                                    onclick: validate_current,
                                    "validate"
                                }
                                button {
                                    disabled: *busy.read(),
                                    class: "primary",
                                    style: "padding: 0.4em 1ch;",
                                    onclick: save_current,
                                    "save"
                                }
                                button {
                                    disabled: *busy.read(),
                                    style: "background: transparent; border: 1px solid var(--error); color: var(--error); padding: 0.4em 1ch;",
                                    onclick: delete_current,
                                    "delete"
                                }
                            }
                            if *busy.read() {
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "working..." }
                            }
                        }

                        div { style: "{panel_style}",
                            span { style: "color: var(--accent);", "validation" }
                            if let Some(result) = validation_value {
                                if result.valid {
                                    div { style: "color: var(--accent);", "Definition is valid." }
                                    textarea {
                                        style: "{input_style} min-height: 12em; font-family: var(--font-mono);",
                                        readonly: true,
                                        value: "{pretty_json(&result.normalized_definition)}",
                                    }
                                } else {
                                    for error in result.errors {
                                        div { style: "color: var(--error);", "{error}" }
                                    }
                                }
                            } else {
                                span { style: "color: var(--muted-color);", "Run validation to see normalized output and schema errors." }
                            }
                        }

                        div { style: "{panel_style}",
                            span { style: "color: var(--accent);", "full definition preview" }
                            textarea {
                                style: "{input_style} min-height: 28em; font-family: var(--font-mono);",
                                readonly: true,
                                value: "{preview_text}",
                            }
                        }
                    }
                }
            }
        }
    }
}
