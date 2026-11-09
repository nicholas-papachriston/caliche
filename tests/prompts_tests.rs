/// Tests for prompt templates and formatting
#[cfg(feature = "prompts")]
mod prompt_tests {
    use caliche::prompts::{ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate};
    use serde_json::json;

    #[test]
    fn test_prompt_template_basic() {
        let template = PromptTemplate::new("Hello, {name}!");
        let result = template
            .format(&json!({"name": "Alice"}))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_prompt_template_multiple_variables() {
        let template = PromptTemplate::new("Translate '{text}' to {language}");
        let result = template
            .format(&json!({
                "text": "Hello",
                "language": "Spanish"
            }))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Translate 'Hello' to Spanish");
    }

    #[test]
    fn test_prompt_template_missing_variable() {
        let template = PromptTemplate::new("Hello, {name}!");
        let result = template.format(&json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing"));
    }

    #[test]
    fn test_prompt_template_extra_variables() {
        let template = PromptTemplate::new("Hello, {name}!");
        let result = template
            .format(&json!({
                "name": "Alice",
                "age": 30,
                "city": "NYC"
            }))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_prompt_template_with_braces() {
        // Template doesn't use {{ }} for escaping, literal braces are preserved
        let template = PromptTemplate::new("Return JSON: {{\"key\": \"{value}\"}}");
        let result = template
            .format(&json!({"value": "test"}))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Return JSON: {{\"key\": \"test\"}}");
    }

    #[test]
    fn test_prompt_template_empty() {
        let template = PromptTemplate::new("");
        let result = template
            .format(&json!({}))
            .expect("Template formatting should succeed");
        assert_eq!(result, "");
    }

    #[test]
    fn test_prompt_template_no_variables() {
        let template = PromptTemplate::new("This is a static template");
        let result = template
            .format(&json!({}))
            .expect("Template formatting should succeed");
        assert_eq!(result, "This is a static template");
    }

    #[test]
    fn test_prompt_template_multiline() {
        let template = PromptTemplate::new("Name: {name}\nAge: {age}\nOccupation: {occupation}");
        let result = template
            .format(&json!({
                "name": "Bob",
                "age": 25,
                "occupation": "Engineer"
            }))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Name: Bob\nAge: 25\nOccupation: Engineer");
    }

    #[test]
    fn test_few_shot_template_basic() {
        use caliche::prompts::Example;

        let examples = vec![
            Example::new("I love this!", "Positive"),
            Example::new("This is terrible.", "Negative"),
        ];

        let template = FewShotPromptTemplate::new(
            examples,
            "Classify the sentiment:",
            "Input: {input}\nOutput:",
        );

        let result = template
            .format(&json!({"input": "This is amazing!"}))
            .expect("Template formatting should succeed");
        assert!(result.contains("Classify the sentiment:"));
        assert!(result.contains("I love this!"));
        assert!(result.contains("Positive"));
        assert!(result.contains("This is terrible."));
        assert!(result.contains("Negative"));
        assert!(result.contains("This is amazing!"));
    }

    #[test]
    fn test_few_shot_template_empty_examples() {
        let template = FewShotPromptTemplate::new(vec![], "Task:", "Input: {input}");
        let result = template
            .format(&json!({"input": "test"}))
            .expect("Template formatting should succeed");
        assert!(result.contains("Task:"));
        assert!(result.contains("test"));
    }

    #[test]
    fn test_few_shot_template_custom_format() {
        use caliche::prompts::Example;

        let template = FewShotPromptTemplate::new(
            vec![Example::new("Question 1", "Answer 1")],
            "Q&A:",
            "Q: {question}",
        )
        .with_example_format("Q: {input}\nA: {output}");

        let result = template
            .format(&json!({"question": "Question 2"}))
            .expect("Template formatting should succeed");
        assert!(result.contains("Q: Question 1"));
        assert!(result.contains("A: Answer 1"));
        assert!(result.contains("Q: Question 2"));
    }

    #[test]
    fn test_few_shot_template_many_examples() {
        use caliche::prompts::Example;

        let examples: Vec<Example> = (0..10)
            .map(|i| Example::new(format!("Input {}", i), format!("Output {}", i)))
            .collect();

        let template = FewShotPromptTemplate::new(examples, "Task:", "Input: {input}");
        let result = template
            .format(&json!({"input": "New input"}))
            .expect("Template formatting should succeed");

        for i in 0..10 {
            assert!(result.contains(&format!("Input {}", i)));
            assert!(result.contains(&format!("Output {}", i)));
        }
        assert!(result.contains("New input"));
    }

    #[test]
    fn test_chat_template_basic() {
        use caliche::prompts::ChatMessage;

        let template = ChatPromptTemplate::new(vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi! How can I help?"),
            ChatMessage::user("Tell me a joke."),
        ]);

        let messages = template
            .format(&json!({}))
            .expect("Template formatting should succeed");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].0, "system");
        assert_eq!(messages[1].0, "user");
        assert_eq!(messages[2].0, "assistant");
        assert_eq!(messages[3].0, "user");
    }

    #[test]
    fn test_chat_template_with_variables() {
        use caliche::prompts::ChatMessage;

        let template = ChatPromptTemplate::new(vec![
            ChatMessage::system("You are a {role}."),
            ChatMessage::user("My name is {name}."),
        ]);

        let messages = template
            .format(&json!({
                "role": "translator",
                "name": "Alice"
            }))
            .expect("Template formatting should succeed");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].1, "You are a translator.");
        assert_eq!(messages[1].1, "My name is Alice.");
    }

    #[test]
    fn test_chat_template_missing_variable() {
        use caliche::prompts::ChatMessage;

        let template = ChatPromptTemplate::new(vec![ChatMessage::user("Hello, {name}!")]);

        let result = template.format(&json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_chat_template_empty() {
        let template = ChatPromptTemplate::new(vec![]);
        let messages = template
            .format(&json!({}))
            .expect("Template formatting should succeed");
        assert_eq!(messages.len(), 0);
    }

    #[test]
    fn test_chat_template_builder_pattern() {
        let template = ChatPromptTemplate::with_system("System")
            .add_user("User 1")
            .add_assistant("Assistant 1")
            .add_user("User 2");

        assert_eq!(
            template
                .format(&json!({}))
                .expect("Template formatting should succeed")
                .len(),
            4
        );
    }

    #[test]
    fn test_chat_template_multiple_variables() {
        use caliche::prompts::ChatMessage;

        let template = ChatPromptTemplate::new(vec![
            ChatMessage::system("You are a {role} specializing in {field}."),
            ChatMessage::user("Tell me about {topic}."),
        ]);

        let messages = template
            .format(&json!({
                "role": "teacher",
                "field": "mathematics",
                "topic": "calculus"
            }))
            .expect("Template formatting should succeed");

        assert_eq!(messages.len(), 2);
        assert!(messages[0].1.contains("teacher"));
        assert!(messages[0].1.contains("mathematics"));
        assert!(messages[1].1.contains("calculus"));
    }

    #[test]
    fn test_prompt_template_special_characters() {
        let template = PromptTemplate::new("Say: {phrase}");
        let result = template
            .format(&json!({
                "phrase": "Hello, {world}! How are you?"
            }))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Say: Hello, {world}! How are you?");
    }

    #[test]
    fn test_prompt_template_unicode() {
        let template = PromptTemplate::new("Translate '{text}' to {language}");
        let result = template
            .format(&json!({
                "text": "こんにちは",
                "language": "English"
            }))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Translate 'こんにちは' to English");
    }

    #[test]
    fn test_few_shot_template_with_separator() {
        use caliche::prompts::Example;

        let template = FewShotPromptTemplate::new(
            vec![Example::new("Good", "Positive")],
            "Classify:",
            "Input: {input}",
        )
        .with_separator("\n---\n");

        let result = template
            .format(&json!({"input": "Great!"}))
            .expect("Template formatting should succeed");
        assert!(result.contains("---"));
    }

    #[test]
    fn test_prompt_template_partial_variables() {
        let mut template = PromptTemplate::new("Translate '{text}' to {language}");
        template = template.with_partial("language", "Spanish");

        let result = template
            .format(&json!({"text": "Hello"}))
            .expect("Template formatting should succeed");
        assert!(result.contains("Spanish"));
    }

    #[test]
    fn test_chat_template_with_system() {
        let template =
            ChatPromptTemplate::with_system("You are an AI assistant.").add_user("Help me");

        let messages = template
            .format(&json!({}))
            .expect("Template formatting should succeed");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].0, "system");
        assert_eq!(messages[0].1, "You are an AI assistant.");
    }

    #[test]
    fn test_few_shot_template_without_prefix() {
        use caliche::prompts::Example;

        let template = FewShotPromptTemplate::without_prefix(
            vec![Example::new("Test", "Result")],
            "Input: {input}",
        );

        let result = template
            .format(&json!({"input": "New"}))
            .expect("Template formatting should succeed");
        assert!(result.contains("Test"));
        assert!(result.contains("New"));
    }
}
