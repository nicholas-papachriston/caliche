//! Rule matching. Matchers answer "does this rule see the request?"
//! They never return an effect.

use crate::action::{Action, ToolFacts};
use crate::credential::CredentialClass;
use crate::effect::Effect;
use crate::paths::{classify_ssrf_host, is_vendor_auth_path};
use crate::posture::ShellDialect;
use crate::request::Request;
use crate::rule::{Matcher, Rule};

pub fn rule_matches(rule: &Rule, request: &Request<'_>) -> bool {
    match &rule.matcher {
        Matcher::ToolClass(class) => request.action.tool_class() == Some(*class),
        Matcher::ToolName(name) => request.action.tool_name() == Some(name.as_str()),
        Matcher::NonHumanApproval => {
            matches!(request.action, Action::ApprovalAnswer { .. }) && !request.principal.is_human()
        }
        Matcher::VendorAuth => touches_vendor_auth(request),
        Matcher::VendorSubscription => {
            matches!(
                request.action,
                Action::ModelUse {
                    cred: CredentialClass::VendorSubscription,
                    ..
                }
            )
        }
        Matcher::Ssrf => is_ssrf(request),
        Matcher::CheckerError => is_checker_error(request),
        Matcher::ShellContent { dialect } => shell_content_matches(*dialect, rule.effect, request),
    }
}

pub fn best_match<'a>(rules: &'a [Rule], request: &Request<'_>) -> Option<&'a Rule> {
    let mut best: Option<&Rule> = None;
    let mut best_specificity = 0_u8;
    for rule in rules {
        if !rule_matches(rule, request) {
            continue;
        }
        let specificity = rule.matcher.specificity();
        // Deny wins inside the layer even when a broader matcher is less specific.
        let replace = best.is_none_or(|previous| {
            rule.effect > previous.effect
                || (rule.effect == previous.effect && specificity > best_specificity)
        });
        if replace {
            best = Some(rule);
            best_specificity = specificity;
        }
    }
    best
}

pub const fn shell_plan_opaque(request: &Request<'_>) -> bool {
    match &request.action {
        Action::Tool {
            facts: ToolFacts::Shell {
                plan: Some(plan), ..
            },
            ..
        } => plan.opaque,
        _ => false,
    }
}

pub fn content_rule_matched(rules: &[Rule], request: &Request<'_>) -> bool {
    rules
        .iter()
        .any(|rule| rule.matcher.is_shell_content() && rule_matches(rule, request))
}

fn shell_content_matches(dialect: ShellDialect, effect: Effect, request: &Request<'_>) -> bool {
    let Action::Tool { facts, .. } = &request.action else {
        return false;
    };
    match facts {
        ToolFacts::Shell {
            dialect: got,
            plan: Some(plan),
        } if *got == dialect => {
            if plan.denied && effect == Effect::Deny {
                true
            } else if plan.opaque {
                effect != Effect::Allow
            } else {
                false
            }
        }
        ToolFacts::CheckerError { .. } => effect != Effect::Allow,
        _ => false,
    }
}

fn touches_vendor_auth(request: &Request<'_>) -> bool {
    match &request.action {
        Action::Tool { facts, .. } => facts_touch_vendor(facts),
        Action::ConfigWrite { target } => is_vendor_auth_path(target),
        _ => false,
    }
}

fn facts_touch_vendor(facts: &ToolFacts) -> bool {
    match facts {
        ToolFacts::Fs { paths, .. } | ToolFacts::Patch { paths } => paths
            .iter()
            .any(|path| is_vendor_auth_path(&path.canonical)),
        ToolFacts::Keychain { owned_by_caliche } => !owned_by_caliche,
        _ => false,
    }
}

fn is_ssrf(request: &Request<'_>) -> bool {
    let host = match &request.action {
        Action::Egress { host, .. }
        | Action::Tool {
            facts: ToolFacts::Http { host, .. },
            ..
        } => host.as_str(),
        _ => return false,
    };
    classify_ssrf_host(host).is_some()
}

const fn is_checker_error(request: &Request<'_>) -> bool {
    match &request.action {
        Action::Unresolved { .. } => true,
        Action::Tool {
            facts: ToolFacts::CheckerError { .. },
            ..
        } => true,
        _ => false,
    }
}
