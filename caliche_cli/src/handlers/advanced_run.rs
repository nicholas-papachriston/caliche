//! Named helpers for `run_advanced_features`.

use super::{AdvancedFeaturesArgs, run_with_cancellation};
use crate::cancellation::CancellationToken;
use anyhow::Result;
use caliche::AgentService;
use caliche::loaders::Document;
use caliche::rag::{RagFusionMode, RagSearchMode, RagService};
use std::sync::Arc;

fn check_cancelled(cancellation_token: Option<&CancellationToken>, stage: &str) -> Result<()> {
    if let Some(token) = cancellation_token
        && token.is_cancelled()
    {
        anyhow::bail!("Operation cancelled by user before {stage}");
    }
    Ok(())
}

fn required_message<'a>(query_text: Option<&'a String>, purpose: &str) -> Result<&'a str> {
    query_text
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("Message required for {purpose}"))
}

async fn run_reasoning_feature(
    args: &AdvancedFeaturesArgs,
    service: &mut AgentService,
    cancellation_token: Option<&CancellationToken>,
    query_text: Option<&String>,
) -> Result<bool> {
    let Some(pattern) = args.reasoning_pattern.as_ref() else {
        return Ok(false);
    };
    use crate::reasoning::*;
    use caliche::config::Config;

    let message = required_message(
        query_text,
        "reasoning pattern (use -m/--message, -p/--prompt, or a trailing prompt)",
    )?;
    if let Some(config_path) = &args.config {
        Config::from_file(config_path)?;
    } else {
        let _ = Config::default();
    }
    let config = Config::default();
    let max_iterations = args
        .max_iterations
        .or(config.reasoning.max_iterations)
        .unwrap_or(5);
    let pattern_owned = pattern.clone();
    let message_owned = message.to_string();
    let tree_depth = args.tree_depth.or(config.reasoning.tree_depth);
    let tree_branches = args.tree_branches.or(config.reasoning.tree_branches);
    let service_clone = service.clone_for_request();
    let result = run_with_cancellation(cancellation_token, async move {
        execute_reasoning_pattern(
            &pattern_owned,
            &message_owned,
            max_iterations,
            tree_depth,
            tree_branches,
            service_clone,
        )
        .await
    })
    .await?;
    display_reasoning_result(&result, args.verbose);
    Ok(true)
}

async fn run_chain_feature(
    args: &AdvancedFeaturesArgs,
    service: &mut AgentService,
    cancellation_token: Option<&CancellationToken>,
    query_text: Option<&String>,
) -> Result<bool> {
    let Some(chain_type) = args.chain_type.as_ref() else {
        return Ok(false);
    };
    use crate::chains::*;
    let message = required_message(
        query_text,
        "chain execution (use -m/--message, -p/--prompt, or a trailing prompt)",
    )?;
    let chain_type_str = chain_type.to_string();
    let template = args.template.clone();
    let template_vars = args.template_vars.clone();
    let message_owned = message.to_string();
    let service_clone = service.clone_for_request();
    let result = run_with_cancellation(cancellation_token, async move {
        execute_chain(
            &chain_type_str,
            template,
            template_vars,
            &message_owned,
            service_clone,
        )
        .await
    })
    .await?;
    display_chain_result(&result);
    Ok(true)
}

fn rag_query_flags(
    args: &AdvancedFeaturesArgs,
) -> (RagSearchMode, RagFusionMode, Option<bool>, Option<bool>) {
    let search_mode =
        crate::documents::rag_search_mode(args.search_mode.as_ref(), RagSearchMode::Dense);
    let fusion_mode = crate::documents::rag_fusion_mode(args.fusion_mode.as_ref());
    let use_mmr = if args.rag_mmr { Some(true) } else { None };
    let use_temporal = if args.rag_temporal_decay {
        Some(true)
    } else {
        None
    };
    (search_mode, fusion_mode, use_mmr, use_temporal)
}

async fn run_document_query(
    args: &AdvancedFeaturesArgs,
    collection: &str,
    query: &str,
    rag_service: Arc<RagService>,
) -> Result<()> {
    use crate::documents::*;
    let top_k = args.rag_top_k.unwrap_or(5);
    let min_similarity = args.rag_min_similarity.unwrap_or(0.0);
    let (search_mode, fusion_mode, use_mmr, use_temporal) = rag_query_flags(args);
    let (results, warnings) = query_documents(
        rag_service,
        query,
        top_k,
        min_similarity,
        search_mode,
        fusion_mode,
        args.dense_weight,
        args.sparse_weight,
        use_mmr,
        args.rag_mmr_lambda,
        use_temporal,
        args.rag_temporal_decay_half_life,
    )
    .await?;
    display_query_result(
        collection,
        query,
        search_mode,
        fusion_mode,
        &results,
        &warnings,
    );
    Ok(())
}

async fn run_document_feature(
    args: &AdvancedFeaturesArgs,
    service: &mut AgentService,
    query_text: Option<&String>,
) -> Result<bool> {
    let Some(doc_path) = args.document.as_ref() else {
        return Ok(false);
    };
    use crate::documents::*;
    let format = args
        .doc_format
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Document format required with --document"))?;
    let documents = load_documents(doc_path, format).await?;
    let chunks = if args.chunk_size > 0 {
        split_documents(documents, args.chunk_size, args.chunk_overlap)
    } else {
        documents
    };
    display_documents(&chunks, Some(5));
    if let (Some(index_collection), Some(query_collection)) =
        (&args.index_to, &args.query_collection)
        && index_collection != query_collection
    {
        anyhow::bail!(
            "--index-to and --query-collection must match when used together in the same CLI invocation"
        );
    }
    if let Some(collection) = args.index_to.as_ref().or(args.query_collection.as_ref()) {
        index_and_maybe_query_documents(args, service, collection, chunks, query_text).await?;
    }
    Ok(true)
}

async fn index_and_maybe_query_documents(
    args: &AdvancedFeaturesArgs,
    service: &mut AgentService,
    collection: &str,
    chunks: Vec<Document>,
    query_text: Option<&String>,
) -> Result<()> {
    use crate::documents::*;
    let rag_tool = Box::pin(build_rag_tool_for_collection(
        service,
        collection,
        args.hashline,
        args.hashline_format,
    ))
    .await?;
    let ids = index_documents(&rag_tool, collection, chunks).await?;
    if args.index_to.is_some() {
        display_indexing_result(collection, &ids);
    }
    if args.query_collection.is_some() {
        let query = query_text
            .map(String::as_str)
            .ok_or_else(|| anyhow::anyhow!("Message required with --query-collection"))?;
        run_document_query(args, collection, query, rag_tool.service()).await?;
    }
    Ok(())
}

async fn run_query_collection_feature(
    args: &AdvancedFeaturesArgs,
    service: &mut AgentService,
    query_text: Option<&String>,
) -> Result<bool> {
    let Some(collection) = args.query_collection.as_ref() else {
        return Ok(false);
    };
    let query = query_text
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("Message required with --query-collection"))?;
    let rag_service = if std::env::var("VECTOR_BACKEND").is_ok() {
        Box::pin(crate::documents::build_rag_tool_for_collection(
            service,
            collection,
            args.hashline,
            args.hashline_format,
        ))
        .await?
        .service()
    } else {
        service.rag_service().ok_or_else(|| {
            anyhow::anyhow!(
                "--query-collection requires either VECTOR_BACKEND for a persistent backend, --document in the same invocation, or a CLI service with RAG already attached"
            )
        })?
    };
    run_document_query(args, collection, query, rag_service).await?;
    Ok(true)
}

async fn run_multi_agent_feature(
    args: &AdvancedFeaturesArgs,
    service: &mut AgentService,
    cancellation_token: Option<&CancellationToken>,
) -> Result<bool> {
    let Some(pattern) = args.multi_agent_pattern.as_ref() else {
        return Ok(false);
    };
    use crate::orchestration::*;
    let message = args
        .message
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Message required for multi-agent pattern"))?;
    let num_agents = 3;
    let pattern_owned = pattern.clone();
    let message_owned = message.clone();
    let service_clone = service.clone_for_request();
    let results = run_with_cancellation(cancellation_token, async move {
        execute_multi_agent_pattern(&pattern_owned, &message_owned, num_agents, service_clone).await
    })
    .await?;
    display_multi_agent_results(pattern, &results);
    Ok(true)
}

async fn run_workflow_feature(args: &AdvancedFeaturesArgs) -> Result<bool> {
    let Some(workflow_path) = args.workflow.as_ref() else {
        return Ok(false);
    };
    use crate::orchestration::*;
    use caliche::llm::ProviderType;
    let config = OrchestrationConfig {
        provider: args
            .provider
            .as_deref()
            .and_then(|value| value.parse::<ProviderType>().ok()),
        model: args.model.clone(),
        tools: args.tools.as_ref().map(|tools| {
            tools
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        }),
    };
    load_workflow(workflow_path, config).await?;
    Ok(true)
}

pub(super) async fn run_advanced_features_impl(
    args: AdvancedFeaturesArgs,
    service: &mut AgentService,
    cancellation_token: Option<&CancellationToken>,
) -> Result<()> {
    let query_text = args.effective_message();

    check_cancelled(cancellation_token, "reasoning pattern")?;
    if run_reasoning_feature(&args, service, cancellation_token, query_text.as_ref()).await? {
        return Ok(());
    }

    check_cancelled(cancellation_token, "chain execution")?;
    if run_chain_feature(&args, service, cancellation_token, query_text.as_ref()).await? {
        return Ok(());
    }

    check_cancelled(cancellation_token, "document processing")?;
    if run_document_feature(&args, service, query_text.as_ref()).await? {
        return Ok(());
    }

    if run_query_collection_feature(&args, service, query_text.as_ref()).await? {
        return Ok(());
    }

    check_cancelled(cancellation_token, "multi-agent dispatch")?;
    if run_multi_agent_feature(&args, service, cancellation_token).await? {
        return Ok(());
    }

    check_cancelled(cancellation_token, "workflow execution")?;
    let _ = run_workflow_feature(&args).await?;
    Ok(())
}
