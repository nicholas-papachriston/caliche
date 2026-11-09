use std::sync::Arc;

impl super::Agent {
    /// Apply a skill to this agent, modifying it in place by:
    /// - Appending the skill's system prompt addition
    /// - Registering the skill's hooks
    /// - Optionally restricting tools to only those required by the skill
    /// - Applying any `max_iterations` override
    ///
    /// Prefer [`Self::register_skill_index`] for discovered on-disk skills so the model
    /// loads full instructions via `fs` only when needed.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use caliche::agent::skills::prebuilt;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// # use std::sync::Arc;
    /// # let llm = Arc::new(caliche::llm::OpenAIProvider::try_new("key".to_string(), "model".to_string())?);
    /// # let tools = Arc::new(caliche::ToolRegistry::new());
    /// # let mut agent = caliche::agent::core::Agent::new(llm, tools);
    /// agent.apply_skill(&prebuilt::code_review());
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply_skill(&mut self, skill: &crate::agent::skills::Skill) {
        tracing::info!("Applying skill: {} - {}", skill.name, skill.description);

        if let Some(ref prompt_addition) = skill.system_prompt_addition {
            self.append_to_system_prompt(prompt_addition);
        }

        for (matcher, hook) in skill.hooks() {
            self.hooks.register(matcher.clone(), Arc::clone(hook));
        }

        if let Some(max_iter) = skill.max_iterations {
            self.set_max_iterations(max_iter);
        }

        if skill.restrict_tools && !skill.required_tools.is_empty() {
            let allowed: Vec<String> = skill.required_tools.iter().cloned().collect();
            self.set_allowed_tools(Some(allowed));
        }
    }

    /// Apply multiple skills to this agent.
    pub fn apply_skills(&mut self, skills: &[&crate::agent::skills::Skill]) {
        for skill in skills {
            self.apply_skill(skill);
        }
    }

    /// Register a skill for progressive disclosure (index only; body loaded via `fs`).
    pub fn register_skill_index(&mut self, entry: crate::agent::skills::SkillIndexEntry) {
        if self
            .skill_index
            .iter()
            .any(|existing| existing.name == entry.name)
        {
            return;
        }
        tracing::info!(
            "Indexed skill for progressive disclosure: {} ({})",
            entry.name,
            entry.location
        );
        self.skill_index.push(entry);
        *self.cached_system_messages.write() = None;
        #[cfg(feature = "tools-rag")]
        {
            *self.cached_system_messages_with_workspace.write() = None;
        }
    }

    /// Register discovered skills that have an on-disk location into the progressive index.
    pub fn register_discovered_skills_index(&mut self, skills: &[crate::agent::skills::Skill]) {
        for skill in skills {
            if let Some(entry) = crate::agent::skills::SkillIndexEntry::from_skill(skill) {
                self.register_skill_index(entry);
            }
        }
    }

    /// Return the progressive skill index.
    #[must_use]
    pub fn skill_index(&self) -> &[crate::agent::skills::SkillIndexEntry] {
        &self.skill_index
    }

    /// Return the current system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}
