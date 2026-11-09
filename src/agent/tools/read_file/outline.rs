use crate::agent::tools::hashline;
use anyhow::Result;

const CODE_SUFFIXES_A: &[&str] = &[
    ".rs", ".py", ".js", ".ts", ".tsx", ".go", ".java", ".c", ".cpp",
];
const CODE_SUFFIXES_B: &[&str] = &[
    ".h", ".json", ".jsonc", ".toml", ".yaml", ".yml", ".html", ".htm",
];

const RUST_STRUCTURE_PREFIXES: &[&str] = &[
    "pub fn ",
    "fn ",
    "pub struct ",
    "struct ",
    "pub enum ",
    "enum ",
    "impl ",
    "pub trait ",
    "mod ",
    "pub mod ",
];
const OTHER_STRUCTURE_PREFIXES: &[&str] = &[
    "def ",
    "class ",
    "async def ",
    "function ",
    "export function ",
    "export class ",
    "export const ",
    "func ",
    "type ",
];

pub(super) fn structural_outline(
    lines: &[&str],
    path: &str,
    file_size: u64,
    total_lines: usize,
) -> Result<String> {
    let mut output = format!(
        "[File: {path}, Size: {file_size} bytes, {total_lines} lines]\n[File too large for full read - showing structure]\n\n"
    );

    if is_code_path(path) {
        output.push_str("## Code Structure\n\n");
        if try_append_code_outline(&mut output, lines, path) {
            return Ok(output);
        }
    } else {
        append_file_preview(&mut output, lines, total_lines);
    }

    output.push_str(&format!(
        "\n[Follow-up: use op=read path=\"{path}\" start_line=N end_line=M, or op=read path=\"{path}\" query=\"your search terms\"]"
    ));

    Ok(output)
}

fn is_code_path(path: &str) -> bool {
    has_suffix(path, CODE_SUFFIXES_A) || has_suffix(path, CODE_SUFFIXES_B)
}

fn has_suffix(path: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| path.ends_with(suffix))
}

fn try_append_code_outline(
    output: &mut String,
    lines: &[&str],
    #[cfg_attr(
        not(feature = "ast"),
        expect(unused_variables, reason = "read by the ast outline")
    )]
    path: &str,
) -> bool {
    #[cfg(feature = "ast")]
    {
        if try_append_ast_outline(output, lines, path) {
            return true;
        }
    }
    append_heuristic_outline(output, lines);
    false
}

#[cfg(feature = "ast")]
fn try_append_ast_outline(output: &mut String, lines: &[&str], path: &str) -> bool {
    let source = lines.join("\n");
    let path_buf = std::path::PathBuf::from(path);
    let language = crate::ast::detect_language(&path_buf);
    if crate::ast::LanguageRegistry::get_language(language).is_none() {
        return false;
    }
    let Ok(ast) = crate::ast::AstParser::parse_source(&source, language, &path_buf) else {
        return false;
    };
    let symbols = crate::ast::extract_symbols(&ast);
    for s in &symbols {
        let display_line = truncate_display(&s.name);
        let line_num = s.line_start + 1;
        let hash = if line_num <= lines.len() {
            hashline::compute_line_hash(lines[line_num - 1])
        } else {
            "??".to_string()
        };
        output.push_str(&format!(
            "  {:>5}:{}: {} {}\n",
            line_num,
            hash,
            s.kind.as_str(),
            display_line
        ));
    }
    if symbols.is_empty() {
        return false;
    }
    output.push_str(&format!(
        "\n[Follow-up: use op=read path=\"{path}\" start_line=N end_line=M, or op=read path=\"{path}\" query=\"your search terms\"]"
    ));
    true
}

fn append_heuristic_outline(output: &mut String, lines: &[&str]) {
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !is_structure_line(trimmed) {
            continue;
        }
        let display_line = truncate_display(trimmed);
        let hash = hashline::compute_line_hash(line);
        output.push_str(&format!("  {:>5}:{hash}: {display_line}\n", idx + 1));
    }
}

fn is_structure_line(trimmed: &str) -> bool {
    has_prefix(trimmed, RUST_STRUCTURE_PREFIXES)
        || has_prefix(trimmed, OTHER_STRUCTURE_PREFIXES)
        || (trimmed.starts_with("const ") && trimmed.contains(" = function"))
}

fn has_prefix(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| text.starts_with(prefix))
}

fn truncate_display(text: &str) -> String {
    if text.len() > 100 {
        format!("{}...", caliche_contracts::text::truncate_utf8(text, 100))
    } else {
        text.to_string()
    }
}

fn append_file_preview(output: &mut String, lines: &[&str], total_lines: usize) {
    output.push_str("## File Preview\n\n");
    output.push_str("### First 20 lines:\n");
    for (idx, line) in lines.iter().take(20).enumerate() {
        let hash = hashline::compute_line_hash(line);
        output.push_str(&format!("  {:>5}:{hash}| {line}\n", idx + 1));
    }

    if total_lines <= 40 {
        return;
    }
    output.push_str(&format!(
        "\n... ({} lines omitted) ...\n\n",
        total_lines - 40
    ));
    output.push_str("### Last 20 lines:\n");
    let start = total_lines - 20;
    for (idx, line) in lines.iter().skip(start).enumerate() {
        let hash = hashline::compute_line_hash(line);
        output.push_str(&format!("  {:>5}:{hash}| {line}\n", start + idx + 1));
    }
}
