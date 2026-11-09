use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permission {
    pub resource: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub permissions: Vec<Permission>,
}

pub struct RbacEnforcer {
    roles: Arc<DashMap<String, Role>>,
}

impl RbacEnforcer {
    pub fn new() -> Self {
        Self {
            roles: Arc::new(DashMap::new()),
        }
    }

    pub fn add_role(&self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    pub fn check_permission(&self, user_roles: &[String], resource: &str, action: &str) -> bool {
        for role_name in user_roles {
            if let Some(role) = self.roles.get(role_name)
                && role
                    .permissions
                    .iter()
                    .any(|p| p.resource == resource && p.action == action)
            {
                return true;
            }
        }
        false
    }
}

impl Default for RbacEnforcer {
    fn default() -> Self {
        Self::new()
    }
}
