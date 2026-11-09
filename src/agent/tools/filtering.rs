use crate::agent::tools::metadata::ToolMetadata;
use crate::agent::tools::selection::SelectionConstraints;

#[must_use]
pub fn passes_constraints(tool: &ToolMetadata, constraints: &SelectionConstraints) -> bool {
    if constraints.excluded_tools.contains(&tool.name) {
        return false;
    }

    if !constraints.required_categories.is_empty()
        && !constraints.required_categories.contains(&tool.category)
    {
        return false;
    }

    if constraints.avoid_side_effects && tool.dependencies.produces_side_effects {
        return false;
    }

    true
}
