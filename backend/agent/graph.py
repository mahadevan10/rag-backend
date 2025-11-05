"""LangGraph workflow wiring for the Vegah agent."""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

from langgraph.graph import END, StateGraph

from .state import AgentState
from ..tools import AgentTools

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return len(text) // 4


def calculate_context_tokens(chunks: list) -> int:
    """Calculate total tokens in context chunks."""
    total = 0
    for chunk in chunks:
        if isinstance(chunk, dict):
            content = chunk.get('content', '')
            total += estimate_tokens(str(content))
    return total


def build_langgraph_agent(agent_tools: AgentTools) -> Callable:
    """Compile the LangGraph agent using the provided tool set."""
    graph = StateGraph(AgentState)

    @graph.add_node
    def analyze_query(state: AgentState) -> dict:
        """
        Advanced query analysis with decomposition for complex queries.
        Breaks down multi-part questions into sub-queries for better retrieval.
        """
        query_lower = state.query.lower()
        
        # Check for page-specific queries
        has_page_number = any(word.startswith("page") or word.isdigit() for word in query_lower.split())
        
        # Extract document names if mentioned
        try:
            all_docs = agent_tools.list_all_documents()
            mentioned_docs = [doc for doc in all_docs if doc.lower() in query_lower]
        except Exception as e:
            logger.warning(f"Could not list documents during analysis: {e}")
            mentioned_docs = []
        
        # Detect complex queries that need decomposition
        complexity_indicators = [
            "compare", "difference", "contrast", "versus", "vs",
            "both", "all", "multiple", "several",
            "and also", "in addition", "furthermore",
            "first", "second", "third", "then"
        ]
        is_complex = any(indicator in query_lower for indicator in complexity_indicators)
        
        # Determine general intent category
        if has_page_number and mentioned_docs:
            intent = "page_specific"
        elif mentioned_docs and len(mentioned_docs) > 1:
            intent = "multi_document_comparison"
        elif mentioned_docs:
            intent = "document_specific"
        elif any(word in query_lower for word in ["summary", "summarize", "overview"]):
            intent = "summarization"
        elif any(word in query_lower for word in ["list", "show", "what", "which", "available"]):
            intent = "listing"
        elif is_complex:
            intent = "complex_multi_part"
        else:
            intent = "general_search"
        
        # Generate sub-queries for complex questions
        sub_queries = []
        if is_complex and len(state.query.split()) > 10:
            # Use LLM to decompose complex query
            try:
                decomposition_prompt = f"""
Break down this complex question into 2-4 simpler sub-questions that can be answered independently.

Complex Question: {state.query}

Respond with ONLY the sub-questions, one per line, numbered:
1. [sub-question]
2. [sub-question]
etc.
"""
                response = agent_tools.llm.generate(
                    decomposition_prompt,
                    "You are a query decomposition expert. Break complex questions into simple parts.",
                    temperature=0.1
                ).strip()
                
                # Parse sub-queries
                for line in response.split('\n'):
                    line = line.strip()
                    if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                        sub_q = line.split('.', 1)[1].strip()
                        if sub_q:
                            sub_queries.append(sub_q)
                
                logger.info(f"📊 Decomposed query into {len(sub_queries)} sub-queries")
            except Exception as e:
                logger.warning(f"Query decomposition failed: {e}")
        
        return {
            **state.model_dump(),
            "intent": intent,
            "key_entities": mentioned_docs,
            "expanded_queries": sub_queries,  # Store sub-queries for later use
            "reasoning": state.reasoning + [
                f"Query analysis: intent={intent}, mentioned_docs={mentioned_docs}, sub_queries={len(sub_queries)}"
            ]
        }

    @graph.add_node
    def list_documents(state: AgentState) -> dict:
        """Retrieve all available documents."""
        try:
            documents = agent_tools.list_all_documents()
            
            # Format document list for context
            doc_list_str = "; ".join(documents) if documents else "No documents available"
            
            return {
                **state.model_dump(),
                "context": state.context + [{
                    "content": f"Available documents: {doc_list_str}",
                    "source": "list_documents"
                }],
                "tools_used": state.tools_used + ["list_all_documents"],
                "reasoning": state.reasoning + [f"Docs: {doc_list_str}"],
            }
        except Exception as e:
            logger.error(f"list_documents failed: {e}")
            return {
                **state.model_dump(),
                "reasoning": state.reasoning + [f"Error listing documents: {str(e)}"],
            }

    @graph.add_node
    def select_tool(state: AgentState) -> dict:
        """Select tool with history awareness."""
        
        # Define allowed tools
        allowed_tools = {
            "summarize_document",
            "search_specific_documents",
            "search_specific_page",
            "broad_search_all_documents",
            "list_all_documents",
            "get_document_overview",
            "select_document",
            "expand_query",
            "get_total_pages",
            "extract_table_of_contents",
            "find_figures_and_tables",
            "extract_references",
            "extract_emails",
            "list_all_chunks_metadata",
            "get_chunk_by_metadata",
            "synthesize_answer",
        }
        
        # Track node calls and increment iterations
        node_counts = state.node_call_counts.copy()
        node_counts["select_tool"] = node_counts.get("select_tool", 0) + 1
        current_iterations = state.iterations + 1
        
        # GUARD: If called too many times, force synthesis
        # Allow more iterations for complex multi-part queries (up to 10 tool selections)
        max_tool_selections = 10
        if node_counts["select_tool"] > max_tool_selections or current_iterations >= state.max_iterations:
            logger.warning(f"Iteration limit reached: select_tool called {node_counts['select_tool']} times, iterations={current_iterations}/{state.max_iterations}")
            return {
                **state.model_dump(),
                "next_tool": "synthesize_answer",
                "is_complete": False,  # Let synthesize_answer handle completion
                "iterations": current_iterations,
                "node_call_counts": node_counts,
                "reasoning": state.reasoning + [
                    f"⚠️ Iteration limit reached ({current_iterations}/{state.max_iterations}) - forcing synthesis with available context"
                ]
            }
        
        # Check what tools were already used
        tools_used = set(state.tools_used)
        
        # Get list of available documents from reasoning (set in list_documents)
        available_docs_count = 0
        for reason in state.reasoning:
            if reason.startswith("Docs:"):
                # Parse document count from reasoning
                available_docs_count = reason.count(";") + 1 if ";" in reason else 1
                break
        
        # Build observation from previous tool results
        observations = []
        if state.context:
            observations.append(f"📊 Gathered {len(state.context)} context chunks")
            # Show brief summary of latest context
            if state.context:
                latest = state.context[-1]
                if isinstance(latest, dict):
                    obs_preview = latest.get('content', str(latest))[:100]
                    observations.append(f"Latest: {obs_preview}...")
        if state.reasoning:
            observations.append(f"Previous reasoning: {state.reasoning[-1]}")
        
        observation_text = "\n".join(observations) if observations else "No observations yet"
        
        prompt = f"""
You are an EXPERT ReAct (Reasoning + Acting) agent for advanced document question-answering.

═══════════════════════════════════════════════════════════
QUERY: {state.query}
═══════════════════════════════════════════════════════════

PREVIOUS OBSERVATIONS:
{observation_text}

CURRENT STATE:
- Intent: {state.intent}
- Tools used: {', '.join(state.tools_used[-5:]) if state.tools_used else 'None'} (showing last 5)
- Context collected: {len(state.context)} chunks from {len(set(ctx.get('filename', 'unknown') for ctx in state.context if isinstance(ctx, dict)))} documents
- Context size: {state.current_context_tokens}/{state.max_context_tokens} tokens ({int(state.current_context_tokens/state.max_context_tokens*100)}% full)
- Iteration: {node_counts.get("select_tool", 0)}/{state.max_iterations}
- Sub-queries available: {len(state.expanded_queries)} (from query decomposition)

ALL AVAILABLE TOOLS (you can use ANY of these):
1. list_all_documents - Get all available document filenames
2. select_document - Pick/identify a document (MUST USE BEFORE summarize_document)
3. summarize_document - Get summary of selected document (requires select_document first)
4. search_specific_documents - Search within specific documents by name
5. search_specific_page - Search within a specific page of a document
6. broad_search_all_documents - Search across all documents
7. get_document_overview - Get metadata about a document
8. expand_query - Generate alternative phrasings of the query
9. get_total_pages - Get page count of a document
10. extract_table_of_contents - Extract ToC from a document
11. find_figures_and_tables - Find all figures/tables in a document
12. extract_references - Extract bibliography/references
13. extract_emails - Extract email addresses from documents
14. list_all_chunks_metadata - List metadata for all chunks (optionally filter by filename)
15. get_chunk_by_metadata - Retrieve specific chunk by filename and chunk_id
16. synthesize_answer - Generate final answer from collected context

IMPORTANT WORKFLOWS:
- Summarization: list_all_documents → select_document → summarize_document → synthesize_answer
- Search: broad_search_all_documents → (if insufficient) expand_query → search again → synthesize_answer
- Comparison: Search each item separately → collect context → synthesize_answer
- Specific info: search_specific_documents or search_specific_page → synthesize_answer

DECISION GUIDELINES:
✓ Have 0 chunks? → Use list_all_documents or broad_search
✓ Have 1-5 chunks? → Consider getting more with higher TOP_K (15-30)
✓ Have 6-15 chunks from diverse sources? → Usually enough, but can get more if needed
✓ Have 15+ chunks? → Likely enough context, consider synthesizing
✓ Context >80% full? → Must synthesize soon or reduce TOP_K
✓ Context >95% full? → MUST synthesize_answer now
✓ Complex multi-part query? → Break it down, use higher TOP_K (20-40)
✓ Specific document mentioned? → Use search_specific_documents
✓ Need comprehensive answer? → Use TOP_K 20-40

RESPOND IN THIS EXACT FORMAT:
THOUGHT: [What am I trying to accomplish? What do I know so far?]
REASONING: [Why is this the best next action? What information gap am I filling?]
ACTION: [exact_tool_name]
QUERY: [refined search query or keywords to use with this tool - only for search tools]
DOCUMENTS: [comma-separated list of document names/keywords if searching specific documents]
TOP_K: [number of chunks to retrieve, default 10, can be 5-50 based on need]

**IMPORTANT for QUERY field:**
- For search tools: Provide the SPECIFIC search terms/keywords you want to find (NOT the full user question)
- For search_specific_page: MUST include "page X" where X is the page number you want to search
- For other tools: Write "N/A" or leave blank
- Extract key concepts from the user query
- Focus on what you're actually searching for

**CRITICAL for search_specific_page:**
When comparing multiple pages, you need to call search_specific_page MULTIPLE TIMES with DIFFERENT page numbers in the QUERY field:
- First call: QUERY: "page 3 content", DOCUMENTS: "rag.pdf"
- Second call: QUERY: "page 5 content", DOCUMENTS: "spec.pdf"

**IMPORTANT for DOCUMENTS field:**
- For search_specific_documents: List the documents you want to search (e.g., "doc1.pdf, doc2.pdf")
- For select_document: List ONE document to select
- For broad_search or other tools: Write "N/A" or "all"
- You can specify multiple documents for comparison queries

**IMPORTANT for TOP_K field:**
- How many chunks you want to retrieve (5-50)
- Use 5-10 for simple queries
- Use 15-30 for comprehensive answers or comparisons
- Use 30-50 for complex multi-part questions
- Current context: {len(state.context)} chunks, ~{state.current_context_tokens} tokens (limit: {state.max_context_tokens})
- If context is getting full, use fewer chunks or synthesize answer

Examples:

Example 1 - Summarization:
THOUGHT: User wants summary of "rag" document. I need to identify which document matches "rag" keyword.
REASONING: Should list documents first to see exact filenames, then select the matching one before summarizing.
ACTION: list_all_documents
QUERY: N/A
DOCUMENTS: N/A
TOP_K: 10

Example 2 - Specific search:
THOUGHT: User asks "what methodology is used in the RAG paper". I need to find methodology information.
REASONING: Should search for "methodology" across all documents to find relevant sections.
ACTION: broad_search_all_documents
QUERY: methodology research approach
DOCUMENTS: all
TOP_K: 15

Example 3 - Document-specific search:
THOUGHT: User wants to know about "evaluation metrics in the rag document". I know the document is PRACTICAL-GUIDE-TO-RAG.pdf from previous observations.
REASONING: Should search within that specific document for evaluation metrics.
ACTION: search_specific_documents
QUERY: evaluation metrics performance measurement
DOCUMENTS: PRACTICAL-GUIDE-TO-RAG.pdf
TOP_K: 10

Example 4 - Multi-document comparison:
THOUGHT: User asks "compare the approaches in the RAG paper and the spec document". I need to search both documents.
REASONING: Should search both documents for their approaches/methodologies and gather context for comparison. Need more chunks for comprehensive comparison.
ACTION: search_specific_documents
QUERY: approach methodology framework
DOCUMENTS: PRACTICAL-GUIDE-TO-RAG.pdf, spec.pdf
TOP_K: 25

Example 5 - Multi-page comparison (IMPORTANT!):
THOUGHT: User asks "compare page 3 of rag document and page 5 of spec document". I need to search DIFFERENT pages.
REASONING: I need to call search_specific_page TWICE - once for page 3, once for page 5. First, I'll search page 3 of rag document.
ACTION: search_specific_page
QUERY: page 3 content
DOCUMENTS: PRACTICAL-GUIDE-TO-RAG.pdf
TOP_K: 10
(Then in next iteration, I will search page 5 of spec.pdf with QUERY: page 5 content)
"""
    
        try:
            response = agent_tools.llm.generate(
                prompt,
                "You are a ReAct agent. You MUST respond in the exact format: THOUGHT:, REASONING:, ACTION:. Be systematic and thorough.",
                temperature=0.2,
            ).strip()
            
            # Parse ReAct format: THOUGHT → REASONING → ACTION → QUERY → DOCUMENTS → TOP_K
            next_tool = None
            thought = ""
            reasoning = ""
            refined_query = ""
            target_documents = ""
            requested_top_k = None
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('THOUGHT:'):
                    thought = line.replace('THOUGHT:', '').strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                elif line.startswith('ACTION:'):
                    action_text = line.replace('ACTION:', '').strip()
                    # Clean up the action (remove quotes, extra text)
                    next_tool = action_text.split()[0].strip().lower()
                elif line.startswith('QUERY:'):
                    refined_query = line.replace('QUERY:', '').strip()
                elif line.startswith('DOCUMENTS:'):
                    target_documents = line.replace('DOCUMENTS:', '').strip()
                elif line.startswith('TOP_K:'):
                    top_k_text = line.replace('TOP_K:', '').strip()
                    try:
                        requested_top_k = int(top_k_text)
                        # Clamp to reasonable range
                        requested_top_k = max(5, min(50, requested_top_k))
                    except (ValueError, TypeError):
                        requested_top_k = None
            
            # Clean refined_query (remove N/A, empty, etc.)
            if refined_query.lower() in ['n/a', 'none', 'null', '']:
                refined_query = ""
            
            # Clean target_documents
            if target_documents.lower() in ['n/a', 'none', 'null', 'all', '']:
                target_documents = ""
            
            # Log the full ReAct trace
            logger.info("=" * 60)
            logger.info("[ReAct] THOUGHT: %s", thought[:200])
            logger.info("[ReAct] REASONING: %s", reasoning[:200])
            logger.info("[ReAct] ACTION: %s", next_tool)
            logger.info("[ReAct] QUERY: %s", refined_query[:200] if refined_query else "(using original query)")
            logger.info("[ReAct] DOCUMENTS: %s", target_documents if target_documents else "(auto-detect or all)")
            logger.info("[ReAct] TOP_K: %s", requested_top_k if requested_top_k else f"(default: {state.top_k})")
            logger.info("=" * 60)
            
            # Fallback if format parsing failed
            if not next_tool:
                logger.warning("⚠️ ReAct format parsing failed. Raw response: %s", response[:300])
                # Try to extract any tool name from the response
                for tool_name in allowed_tools:
                    if tool_name in response.lower():
                        next_tool = tool_name
                        logger.info("Extracted tool from response: %s", tool_name)
                        break
                
                if not next_tool:
                    logger.warning("No tool found, defaulting to synthesize_answer")
                    next_tool = "synthesize_answer"
            
        except Exception as exc:
            logger.error("❌ ReAct reasoning failed: %s", exc, exc_info=True)
            # Fallback logic based on state
            if len(state.context) == 0:
                next_tool = "broad_search_all_documents"
                logger.info("Fallback: No context, searching all documents")
            else:
                next_tool = "synthesize_answer"
                logger.info("Fallback: Have context, synthesizing answer")

        # Validate tool selection
        if next_tool not in allowed_tools:
            logger.warning("Invalid tool '%s', defaulting to synthesize", next_tool)
            next_tool = "synthesize_answer"

        # Safety: prevent immediate infinite loops (10+ consecutive identical tools)
        # But allow 2 consecutive uses of search tools (e.g., search page 3, then search page 5)
        if len(state.tools_used) >= 10 and state.tools_used[-1] == state.tools_used[-2] == state.tools_used[-3] == state.tools_used[-4] == state.tools_used[-5] == state.tools_used[-6] == state.tools_used[-7] == state.tools_used[-8] == state.tools_used[-9] == state.tools_used[-10]:
            logger.warning("Agent used tool '%s' 10 times in a row, forcing synthesis to prevent infinite loop", next_tool)
            next_tool = "synthesize_answer"
        
        # Force synthesis after max iterations
        if current_iterations >= state.max_iterations:
            logger.info("Max iterations reached, forcing synthesis")
            next_tool = "synthesize_answer"
        
        # Force synthesis if agent is repeating the same search operation WITHOUT refinement
        # Check if the last 2 tools were the same search tool
        if len(state.tools_used) >= 2 and state.tools_used[-1] == state.tools_used[-2]:
            if state.tools_used[-1] in ["search_specific_page", "broad_search_all_documents", "search_specific_documents"]:
                # Allow repeat if:
                # 1. Agent is refining the query (different refined_query)
                # 2. We got insufficient results (≤5 chunks) - might need different search or more chunks
                # 3. It's a complex query that needs comprehensive context
                has_refined_query = refined_query and refined_query != state.query
                has_insufficient_results = len(state.context) <= 7
                is_complex_query = state.intent in ["complex_multi_part", "multi_document_comparison"] or any(
                    word in state.query.lower() for word in ["compare", "both", "all", "each"]
                )
                
                if not (has_refined_query or has_insufficient_results or is_complex_query):
                    logger.info("Agent repeating same search operation (%s) without refinement, forcing synthesis to avoid loop", state.tools_used[-1])
                    next_tool = "synthesize_answer"
                else:
                    logger.info("Agent repeating search - allowing it (refined=%s, insufficient=%s, complex=%s)", 
                                has_refined_query, has_insufficient_results, is_complex_query)
        
        # Force synthesis if we have good context and low query complexity
        # For simple queries with sufficient results, one search should be enough
        if len(state.context) >= 5 and next_tool in ["search_specific_page", "broad_search_all_documents", "search_specific_documents"]:
            # Count how many times we've searched already
            search_count = sum(1 for tool in state.tools_used if tool in ["search_specific_page", "broad_search_all_documents", "search_specific_documents"])
            # For simple queries (no listing/comparison keywords), one successful search is enough
            simple_query = not any(word in state.query.lower() for word in ["and", "compare", "between", "both", "multiple", "all", "each", "list", "what are"])
            if simple_query and search_count >= 1:
                logger.info("Simple query with %d search(es) completed and %d chunks collected, forcing synthesis", search_count, len(state.context))
                next_tool = "synthesize_answer"

        logger.info(
            "[ReAct] action=%s | used=%s | context=%d | iter=%d/%d",
            next_tool,
            state.tools_used[-2:] if len(state.tools_used) >= 2 else state.tools_used,
            len(state.context),
            current_iterations,
            state.max_iterations,
        )

        # Update reasoning with agent's thought process
        new_reasoning = state.reasoning.copy()
        if thought:
            new_reasoning.append(f"💭 {thought}")
        if reasoning:
            new_reasoning.append(f"⚡ {reasoning}")
        new_reasoning.append(f"→ Action: {next_tool}")
        
        # EXTRACT ENTITIES: If tool needs specific documents, try to extract document references
        extracted_entities = []
        extracted_doc_ids = []
        
        if next_tool in ["select_document", "summarize_document", "search_specific_documents", "search_specific_page"]:
            # Priority 1: Parse DOCUMENTS field from agent's response
            if target_documents:
                all_docs = agent_tools.list_all_documents()
                # Split by comma and clean up
                doc_candidates = [d.strip() for d in target_documents.split(',')]
                
                for candidate in doc_candidates:
                    # Try exact match
                    for doc in all_docs:
                        if candidate.lower() == doc.lower():
                            extracted_doc_ids.append(doc)
                            extracted_entities.append(doc)
                            logger.info(f"Exact match from DOCUMENTS field: {doc}")
                            break
                    
                    # Try fuzzy match if no exact match
                    if not extracted_doc_ids:
                        candidate_lower = candidate.lower().replace('.pdf', '')
                        for doc in all_docs:
                            doc_lower = doc.lower().replace('.pdf', '')
                            if candidate_lower in doc_lower or doc_lower in candidate_lower:
                                extracted_doc_ids.append(doc)
                                extracted_entities.append(doc)
                                logger.info(f"Fuzzy match from DOCUMENTS field: {doc} (candidate: {candidate})")
                                break
            
            # Priority 2: Extract from THOUGHT/REASONING text if DOCUMENTS field empty
            if not extracted_entities:
                all_docs = agent_tools.list_all_documents()
                search_text = f"{thought} {reasoning} {state.query}".lower()
                
                # Find ALL exact document name mentions (not just first)
                for doc in all_docs:
                    doc_lower = doc.lower()
                    if doc_lower in search_text:
                        extracted_entities.append(doc)
                        extracted_doc_ids.append(doc)
                        logger.info(f"Extracted entity from agent reasoning: {doc}")
                
                # If no exact match, try partial matching with ALL documents
                if not extracted_entities:
                    for doc in all_docs:
                        doc_parts = doc.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ').split()
                        # Check if any significant part of document name appears in search text
                        for part in doc_parts:
                            if len(part) > 3 and part in search_text:
                                extracted_entities.append(doc)
                                extracted_doc_ids.append(doc)
                                logger.info(f"Fuzzy matched entity from reasoning: {doc} (matched '{part}')")
                                break
                
                # If still no match, extract keywords for fuzzy matching later
                if not extracted_entities:
                    stop_words = {'the', 'a', 'an', 'what', 'does', 'say', 'about', 'is', 'are', 'document', 'file', 'pdf', 'in', 'from', 'of'}
                    words = [w.strip('?.,!') for w in state.query.lower().split() if w.strip('?.,!') not in stop_words and len(w) > 2]
                    extracted_entities = words[:3]  # Take top 3 keywords for fuzzy matching
                    logger.info(f"Extracted keywords for fuzzy matching: {extracted_entities}")
        
        return {
            **state.model_dump(), 
            "next_tool": next_tool, 
            "reasoning": new_reasoning,
            "iterations": current_iterations,
            "node_call_counts": node_counts,
            "key_entities": extracted_entities if extracted_entities else state.key_entities,
            "refined_query": refined_query if refined_query else state.refined_query,  # Store agent's refined search query
            "doc_ids": extracted_doc_ids if extracted_doc_ids else state.doc_ids,  # Store agent's selected documents
            "top_k": requested_top_k if requested_top_k else state.top_k,  # Agent can adjust chunk retrieval
        }

    @graph.add_node
    def expand_query(state: AgentState):
        result = agent_tools.expand_query(state.query)
        expansions = result.get("expanded_queries", [])
        tools_used = state.tools_used + ["expand_query"]
        reasoning = state.reasoning + [f"Expanded queries: {expansions[:3]}"]
        return {
            **state.model_dump(),
            "expanded_queries": expansions or state.expanded_queries,
            "tools_used": tools_used,
            "reasoning": reasoning,
        }

    @graph.add_node
    def select_document(state: AgentState):
        # Try to find the document name from key_entities or by matching
        filename = None
        
        if state.key_entities:
            filename = state.key_entities[0]
        else:
            # Try to match document name from query using smart fuzzy matching
            all_docs = agent_tools.list_all_documents()
            query_lower = state.query.lower()
            
            # Extract potential document keywords from query
            # Remove common words and punctuation
            stop_words = {'the', 'a', 'an', 'what', 'does', 'say', 'about', 'is', 'are', 'document', 'file', 'pdf'}
            query_words = [w.strip('?.,!') for w in query_lower.split() if w.strip('?.,!') not in stop_words]
            
            # Try exact match first
            for doc in all_docs:
                doc_lower = doc.lower()
                if doc_lower in query_lower:
                    filename = doc
                    break
            
            # Try fuzzy matching with document name parts
            if not filename:
                best_match = None
                best_score = 0
                
                for doc in all_docs:
                    doc_parts = doc.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ').split()
                    
                    # Count how many query words appear in document name
                    matches = sum(1 for qw in query_words if any(qw in dp or dp in qw for dp in doc_parts))
                    
                    if matches > best_score:
                        best_score = matches
                        best_match = doc
                
                # Accept match if at least one keyword matched
                if best_score > 0:
                    filename = best_match
                    logger.info(f"Fuzzy matched '{state.query}' to '{filename}' (score: {best_score})")
        
        if not filename:
            error_msg = f"Could not identify which document to select from query: {state.query}"
            logger.warning(error_msg)
            reasoning = state.reasoning + [f"❌ {error_msg}"]
            return {
                **state.model_dump(),
                "reasoning": reasoning,
                "tools_used": state.tools_used + ["select_document"],
            }
        
        result = agent_tools.select_document(filename)
        
        # Handle errors from the tool
        if "error" in result:
            error_msg = result["error"]
            reasoning = state.reasoning + [f"❌ Document selection failed: {error_msg}"]
            return {
                **state.model_dump(),
                "reasoning": reasoning,
                "tools_used": state.tools_used + ["select_document"],
            }
        
        # Store the selected document in doc_ids for other tools to use
        doc_ids = [filename]
        selected_docs = [filename]
        
        # selected_documents is a list of strings (filenames), not dicts
        if isinstance(selected_docs, list) and selected_docs:
            selected_names = [str(doc) for doc in selected_docs if doc]
        else:
            selected_names = []
        
        tools_used = state.tools_used + ["select_document"]
        if selected_names:
            selection_text = ", ".join(selected_names)
        elif doc_ids:
            selection_text = f"{len(doc_ids)} document(s) selected"
        else:
            selection_text = "No documents selected"
        reasoning = state.reasoning + [f"Selected documents: {selection_text}"]
        return {
            **state.model_dump(),
            "doc_ids": doc_ids,
            "tools_used": tools_used,
            "reasoning": reasoning,
        }

    @graph.add_node
    def get_document_overview(state: AgentState):
        doc_identifier = state.key_entities[0] if state.key_entities else "rag"
        result = agent_tools.get_document_overview(doc_identifier)
        tools_used = state.tools_used + ["get_document_overview"]
        if "error" in result:
            overview_text = f"Overview unavailable: {result['error']}"
        else:
            name = result.get("filename", doc_identifier or "Unknown document")
            indexed = result.get("indexed_pages", 0)
            total = result.get("total_pages", 0)
            overview_text = f"Overview: {name} — {indexed}/{total} pages indexed"
        reasoning = state.reasoning + [overview_text]
        return {**state.model_dump(), "tools_used": tools_used, "reasoning": reasoning}

    @graph.add_node
    def summarize_document(state: AgentState):
        # Get filename from doc_ids or key_entities
        filename = None
        if state.doc_ids:
            filename = state.doc_ids[0]
        elif state.key_entities:
            filename = state.key_entities[0]
        
        if not filename:
            logger.warning("No document specified for summarize_document")
            return {
                **state.model_dump(),
                "reasoning": state.reasoning + ["⚠️ No document specified for summary"],
            }
        
        # Check context window limit
        if state.current_context_tokens >= state.max_context_tokens:
            reasoning = state.reasoning + [f"⚠️ Context window full. Skipping summary."]
            return {
                **state.model_dump(),
                "reasoning": reasoning,
            }
        
        result = agent_tools.summarize_document(filename=filename)
        new_chunks = result.get("chunks", [])
        context = state.context + new_chunks
        
        # Update token count
        new_tokens = calculate_context_tokens(new_chunks)
        total_tokens = state.current_context_tokens + new_tokens
        
        tools_used = state.tools_used + ["summarize_document"]
        reasoning = state.reasoning + [f"Summarized: {filename} → +{new_tokens} tokens"]
        return {
            **state.model_dump(),
            "context": context,
            "tools_used": tools_used,
            "reasoning": reasoning,
            "current_context_tokens": total_tokens,
        }

    @graph.add_node
    def search_specific_documents(state: AgentState):
        # Use refined query from agent's reasoning, fallback to original
        search_query = state.refined_query or state.query
        
        # Check context window limit
        if state.current_context_tokens >= state.max_context_tokens:
            reasoning = state.reasoning + [f"⚠️ Context window full ({state.current_context_tokens}/{state.max_context_tokens} tokens). Skipping search."]
            return {
                **state.model_dump(),
                "reasoning": reasoning,
            }
        
        result = agent_tools.search_specific_documents(
            search_query, state.doc_ids, state.top_k
        )
        new_chunks = result.get("chunks", [])
        context = state.context + new_chunks
        
        # Update token count
        new_tokens = calculate_context_tokens(new_chunks)
        total_tokens = state.current_context_tokens + new_tokens
        
        tools_used = state.tools_used + ["search_specific_documents"]
        reasoning = state.reasoning + [f"Searched '{search_query[:50]}' in {len(state.doc_ids)} doc(s) → {len(new_chunks)} chunks, +{new_tokens} tokens"]
        return {
            **state.model_dump(),
            "context": context,
            "tools_used": tools_used,
            "reasoning": reasoning,
            "current_context_tokens": total_tokens,
        }

    @graph.add_node
    def broad_search_all_documents(state: AgentState):
        # Use refined query from agent's reasoning, fallback to original
        search_query = state.refined_query or state.query
        
        # Check context window limit
        if state.current_context_tokens >= state.max_context_tokens:
            reasoning = state.reasoning + [f"⚠️ Context window full ({state.current_context_tokens}/{state.max_context_tokens} tokens). Skipping search."]
            return {
                **state.model_dump(),
                "reasoning": reasoning,
            }
        
        result = agent_tools.broad_search_all_documents(search_query, state.top_k)
        new_chunks = result.get("chunks", [])
        context = state.context + new_chunks
        
        # Update token count
        new_tokens = calculate_context_tokens(new_chunks)
        total_tokens = state.current_context_tokens + new_tokens
        
        tools_used = state.tools_used + ["broad_search_all_documents"]
        reasoning = state.reasoning + [f"Broad search: '{search_query[:50]}' → {len(new_chunks)} chunks, +{new_tokens} tokens (total: {total_tokens})"]
        return {
            **state.model_dump(),
            "context": context,
            "tools_used": tools_used,
            "reasoning": reasoning,
            "current_context_tokens": total_tokens,
        }

    @graph.add_node
    def search_specific_page(state: AgentState):
        # Use refined query for searching
        search_query = state.refined_query or state.query
        
        # Extract page number - prefer from refined_query (agent's intent), fallback to original query
        # This allows agent to specify different pages for multi-page comparisons
        query_to_parse = state.refined_query if state.refined_query else state.query
        match = re.search(r"page\s*(\d+)", query_to_parse.lower())
        if not match:
            # Fallback: try original query
            match = re.search(r"page\s*(\d+)", state.query.lower())
        page_number = int(match.group(1)) if match else 1
        
        logger.info(f"Extracted page number {page_number} from: '{query_to_parse[:100]}'...")
        
        # Strategy 1: Use doc_ids if agent specified them (PREFERRED - agent knows best!)
        if state.doc_ids and len(state.doc_ids) > 0:
            target_filenames = state.doc_ids
            logger.info(f"Using agent-specified documents: {target_filenames}")
        else:
            # Strategy 2: Fallback - detect multi-doc keywords in query
            query_lower = state.query.lower()
            multi_doc_keywords = ["each document", "all documents", "both documents", "both doc"]
            is_multi_doc_query = any(keyword in query_lower for keyword in multi_doc_keywords) or re.search(r"compare.*(both|all)", query_lower)
            
            if is_multi_doc_query:
                target_filenames = agent_tools.list_all_documents()
                logger.info(f"Multi-doc query detected, using all documents: {target_filenames}")
            else:
                target_filenames = None
        
        # Multi-document search (if we have multiple targets)
        if target_filenames and len(target_filenames) > 1:
            all_filenames = target_filenames
            
            # Search the specific page in EACH document
            all_results = []
            for filename in all_filenames:
                logger.info(f"Searching page {page_number} in document: {filename}")
                result = agent_tools.search_specific_page(filename, page_number, search_query)
                chunks = result.get("chunks", [])
                if chunks and "error" not in result:
                    all_results.extend(chunks)
                    logger.info(f"✓ Found {len(chunks)} chunks from {filename} page {page_number}")
                elif "error" in result:
                    logger.warning(f"⚠️ Page {page_number} not found in {filename}")
                    # Add a placeholder indicating page not found in this doc
                    all_results.append({
                        "content": f"Page {page_number} not found in {filename}",
                        "filename": filename,
                        "page": page_number,
                    })
            
            context = state.context + all_results
            doc_list = ", ".join(all_filenames)
            reasoning_msg = f"Searched page {page_number} in all documents ({doc_list}) → {len(all_results)} total chunks"
        else:
            # Single document page search
            # Try to get filename from doc_ids (agent specified), then fallback options
            filename = None
            
            if target_filenames and len(target_filenames) == 1:
                # Agent specified exactly one document
                filename = target_filenames[0]
                logger.info(f"Using agent-specified single document: {filename}")
            elif state.key_entities:
                # Fallback: key_entities contains document names extracted from query
                filename = state.key_entities[0]
                logger.info(f"Using document from key_entities: {filename}")
            
            # If still no filename, try to get the only document if there's just one
            if not filename:
                all_filenames = agent_tools.list_all_documents()
                if len(all_filenames) == 1:
                    filename = all_filenames[0]
                    logger.info(f"Only one document available, using: {filename}")
                else:
                    logger.warning(f"No document specified and {len(all_filenames)} documents available")
            
            result = agent_tools.search_specific_page(
                filename, page_number, search_query
            )
            context = state.context + result.get("chunks", [])
            reasoning_msg = f"Searched page {page_number} in '{filename}' with query: '{search_query[:50]}'"
        
        tools_used = state.tools_used + ["search_specific_page"]
        reasoning = state.reasoning + [reasoning_msg]
        return {
            **state.model_dump(),
            "context": context,
            "tools_used": tools_used,
            "reasoning": reasoning,
        }

    @graph.add_node
    def synthesize_answer(state: AgentState) -> dict:
        """Generate final answer with context-aware prompting."""
        
        # Analyze what context we have
        has_content = False
        has_filenames_only = False
        
        for ctx in state.context:
            ctx_str = str(ctx).lower()
            if any(keyword in ctx_str for keyword in ["sample", "page", "chunk", "content:"]):
                has_content = True
                break
            elif "available documents:" in ctx_str or "document:" in ctx_str:
                has_filenames_only = True
        
        # Build context-aware system prompt with attribution requirements
        if has_filenames_only and not has_content:
            system_prompt = """You are a helpful document assistant. 

IMPORTANT: When you only have document filenames (not content):
1. List the documents clearly and confidently
2. Acknowledge you can provide more details if needed
3. Be proactive - offer to summarize, search, or answer specific questions
4. Don't apologize or say "limited information" - you have the ability to access full content

Example good responses:
- "I have: [filename]. Would you like me to provide a summary of its contents?"
- "The available document is [filename]. I can search through it or summarize it for you."
- "I found [filename]. What would you like to know about it?"

Be confident and helpful!"""
        
        elif has_content:
            system_prompt = """You are an expert document assistant with emphasis on ACCURACY and SOURCE ATTRIBUTION.

CRITICAL REQUIREMENTS:
1. CITE SOURCES: Always mention which document and page number information comes from
2. BE SPECIFIC: Include concrete details, numbers, quotes when available
3. BE ACCURATE: Only state facts that are directly supported by the context
4. BE COMPLETE: Address all parts of the question
5. NO HALLUCINATION: If information is missing, explicitly state what you don't know
6. STRUCTURE: Use clear formatting (bullet points, numbering) for complex answers

⚠️ CRITICAL CITATION RULES:
- ONLY cite page numbers that are EXPLICITLY mentioned in the provided context
- DO NOT invent or guess page numbers
- If the context doesn't include a page number, cite as "[Source: filename]" WITHOUT page number
- If you're unsure about a page number, do NOT include it in the citation
- It's better to omit the page number than to cite an incorrect one

FORMAT YOUR ANSWER:
- Start with direct answer to the question
- Support with specific details from sources
- Cite as: "[Source: filename, page X]" ONLY if page X is explicitly in the context
- If no page number available, cite as: "[Source: filename]"
- End with any limitations or additional context needed

Example CORRECT citations:
"The main methodology is quantitative analysis [Source: paper.pdf, page 5]." (if context shows page 5)
"The methodology uses surveys [Source: paper.pdf]." (if context has no page number)

Example WRONG citations:
"The sample size was N=500 [Source: paper.pdf, page 55]." (if context never mentions page 55)

Remember: Accuracy > Completeness. Better to say "I don't have information about X" than to guess."""
        
        else:
            system_prompt = """You are a helpful document assistant.

You don't have any documents or context available yet.
Politely let the user know and suggest they upload documents first."""
        
        # Build user prompt with context - include explicit page numbers
        context_items = []
        for i, ctx in enumerate(state.context, 1):
            if isinstance(ctx, dict):
                # Extract source info
                metadata = ctx.get('metadata', {})
                source = ctx.get('source') or ctx.get('filename') or metadata.get('filename', 'unknown')
                page = ctx.get('page') or metadata.get('page_number')
                content = ctx.get('content', str(ctx))
                
                # Format with explicit page number if available
                if page is not None:
                    header = f"[Chunk {i}] Source: {source}, Page: {page}"
                else:
                    header = f"[Chunk {i}] Source: {source}"
                
                context_items.append(f"{header}\n{content}")
            else:
                context_items.append(f"[Chunk {i}] {str(ctx)}")
        
        context_text = "\n\n---\n\n".join(context_items) if context_items else "No context available"
        
        # Add reasoning trail for transparency
        reasoning_text = "\n".join([f"- {r}" for r in state.reasoning[-5:]]) if state.reasoning else "No reasoning trail"
        
        user_prompt = f"""User Query: {state.query}

Tools Used: {', '.join(state.tools_used) if state.tools_used else 'None'}

Reasoning Trail:
{reasoning_text}

Retrieved Context:
{context_text}

Based on the above, provide a helpful answer to the user's query."""
        
        try:
            # Generate answer
            answer = agent_tools.llm.generate(
                user_prompt,
                system_prompt,
                temperature=0.3
            )
            
            # Ensure answer is not None or empty
            if not answer or answer.strip() == "":
                answer = "I processed your query but couldn't generate a meaningful answer. Please try rephrasing."
            
            # VALIDATION: Check for hallucinated page citations
            # Extract all page numbers that appear in citations [Source: ..., page X]
            import re
            citation_pattern = r'\[Source:.*?page\s+(\d+)\]'
            cited_pages = set(re.findall(citation_pattern, answer, re.IGNORECASE))
            
            # Get actual available pages from context
            available_pages = set()
            page_ranges = {}  # filename -> (min_page, max_page)
            
            for ctx in state.context:
                if isinstance(ctx, dict):
                    metadata = ctx.get('metadata', {})
                    page = ctx.get('page') or metadata.get('page_number')
                    filename = ctx.get('filename') or metadata.get('filename', 'unknown')
                    total_pages = metadata.get('total_pages')
                    
                    if page is not None:
                        # Ensure page is an integer for comparison
                        try:
                            page_int = int(page)
                        except (ValueError, TypeError):
                            continue  # Skip invalid page numbers
                        
                        available_pages.add(str(page_int))
                        
                        # Track page ranges per document (store as integers)
                        if filename not in page_ranges:
                            page_ranges[filename] = (page_int, page_int, total_pages)
                        else:
                            min_p, max_p, total = page_ranges[filename]
                            page_ranges[filename] = (min(min_p, page_int), max(max_p, page_int), total)
            
            # Check for hallucinated citations
            hallucinated_pages = cited_pages - available_pages
            if hallucinated_pages:
                logger.warning(
                    f"⚠️ HALLUCINATION DETECTED: Answer cites pages {hallucinated_pages} "
                    f"which are NOT in the retrieved context. Available pages: {available_pages}"
                )
                
                # Add warning to reasoning
                warning_msg = (
                    f"⚠️ Citation validation: Answer mentions page(s) {', '.join(sorted(hallucinated_pages))} "
                    f"which were not in retrieved context (available: {', '.join(sorted(available_pages)[:10])})"
                )
                new_reasoning = state.reasoning + [warning_msg]
                
                # Log document ranges for debugging
                for fname, (min_p, max_p, total) in page_ranges.items():
                    logger.info(f"Document '{fname}': pages {min_p}-{max_p} in context (total: {total} pages)")
            else:
                new_reasoning = state.reasoning + ["✅ Citations validated - all page numbers match retrieved context"]
            
            return {
                **state.model_dump(),
                "final_answer": answer.strip(),
                "is_complete": True,
                "reasoning": new_reasoning + ["Generated final answer with available context"]
            }
        
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}", exc_info=True)
            
            # Fallback answer on error
            fallback_answer = f"I encountered an error while processing your query. Please try again or rephrase your question."
            
            return {
                **state.model_dump(),
                "final_answer": fallback_answer,
                "is_complete": True,
                "reasoning": state.reasoning + [f"Error in synthesis: {str(e)}"]
            }

    @graph.add_node
    def review_answer(state: AgentState) -> dict:
        """
        Advanced Review Agent with confidence scoring and self-reflection.
        Validates answer quality, assigns confidence scores, and can request more context.
        """
        
        if not state.final_answer:
            logger.warning("No answer to review, skipping review")
            return {**state.model_dump(), "is_complete": True}
        
        # Count sources and check coverage
        num_sources = len([ctx for ctx in state.context if isinstance(ctx, dict)])
        has_multiple_sources = num_sources > 2
        
        # Build review prompt with confidence scoring
        review_prompt = f"""
You are an ADVANCED REVIEW AGENT with expertise in document QA validation.

ORIGINAL QUERY: {state.query}

ANSWER PROVIDED:
{state.final_answer}

CONTEXT USED ({num_sources} sources):
{chr(10).join([f"- {ctx.get('filename', 'unknown')} (page {ctx.get('page', 'N/A')}): {ctx.get('content', str(ctx))[:150]}..." for ctx in state.context[:7] if isinstance(ctx, dict)])}

TOOLS USED: {', '.join(state.tools_used[-5:])}

EVALUATION CRITERIA:
1. ACCURACY: Does the answer match the source documents? Any hallucinations?
2. COMPLETENESS: Are all parts of the query addressed?
3. SPECIFICITY: Are concrete details/numbers/facts cited?
4. SOURCE COVERAGE: Is the answer based on sufficient sources?
5. CLARITY: Is the answer clear and well-structured?

RESPOND IN THIS EXACT FORMAT:
CONFIDENCE_SCORE: [0-100, where 100 is perfect confidence]
CRITIQUE: [Detailed assessment of strengths and weaknesses]
ACCURACY_ISSUES: [List any factual errors or hallucinations, or "NONE"]
MISSING_INFORMATION: [What's missing from the answer, or "NONE"]
NEEDS_IMPROVEMENT: [YES or NO]

If NEEDS_IMPROVEMENT is YES, you MUST provide a complete revised answer below:
REVISED_ANSWER: [Write the COMPLETE improved answer here, ready to be shown to the user]

If NEEDS_IMPROVEMENT is NO:
REVISED_ANSWER: APPROVED

CRITICAL: The REVISED_ANSWER must be a COMPLETE, STANDALONE answer that can be shown directly to the user.
Do NOT write meta-commentary or suggestions. Write the actual improved answer.

Example BAD (meta-commentary):
REVISED_ANSWER: The answer should include specific information from the document...

Example GOOD (complete answer):
REVISED_ANSWER: Based on the spec.pdf, the main features include: 1) Authentication via OAuth 2.0 [Source: spec.pdf, page 3], 2) Rate limiting of 1000 requests/hour [Source: spec.pdf, page 5]...

Be critical and thorough. Low confidence (<70) should trigger improvement.
"""

        try:
            review_response = agent_tools.llm.generate(
                review_prompt,
                "You are a critical review agent with expertise in quality assurance. Be thorough, honest, and assign accurate confidence scores.",
                temperature=0.15  # Lower temp for more consistent evaluation
            ).strip()
            
            # Parse review response with confidence scoring
            confidence_score = 0
            critique = ""
            accuracy_issues = ""
            missing_info = ""
            needs_improvement = "NO"
            improvement_action = "NONE"
            revised_answer = "APPROVED"
            
            for line in review_response.split('\n'):
                line = line.strip()
                if line.startswith('CONFIDENCE_SCORE:'):
                    try:
                        confidence_score = int(line.replace('CONFIDENCE_SCORE:', '').strip().split()[0])
                    except:
                        confidence_score = 50  # Default to medium
                elif line.startswith('CRITIQUE:'):
                    critique = line.replace('CRITIQUE:', '').strip()
                elif line.startswith('ACCURACY_ISSUES:'):
                    accuracy_issues = line.replace('ACCURACY_ISSUES:', '').strip()
                elif line.startswith('MISSING_INFORMATION:'):
                    missing_info = line.replace('MISSING_INFORMATION:', '').strip()
                elif line.startswith('NEEDS_IMPROVEMENT:'):
                    needs_improvement = line.replace('NEEDS_IMPROVEMENT:', '').strip().upper()
                elif line.startswith('IMPROVEMENT_ACTION:'):
                    improvement_action = line.replace('IMPROVEMENT_ACTION:', '').strip()
                elif line.startswith('REVISED_ANSWER:'):
                    # Get all remaining lines as revised answer
                    idx = review_response.index(line)
                    revised_answer = review_response[idx+len('REVISED_ANSWER:'):].strip()
                    break
            
            logger.info("=" * 70)
            logger.info("[Review] CONFIDENCE: %d/100", confidence_score)
            logger.info("[Review] CRITIQUE: %s", critique[:200])
            logger.info("[Review] ACCURACY_ISSUES: %s", accuracy_issues[:100])
            logger.info("[Review] MISSING_INFO: %s", missing_info[:100])
            logger.info("[Review] NEEDS_IMPROVEMENT: %s", needs_improvement)
            if revised_answer not in ["APPROVED", ""]:
                logger.info("[Review] REVISED_ANSWER (first 200 chars): %s", revised_answer[:200])
            logger.info("=" * 70)
            
            # Decision logic based on confidence and improvements
            if confidence_score < 70:
                logger.warning(f"⚠️ Low confidence ({confidence_score}/100) - checking for revised answer")
                needs_improvement = "YES"
            
            # Determine final answer
            if needs_improvement == "YES" and revised_answer not in ["APPROVED", "", "NONE"]:
                # Check if revised_answer is actually a complete answer (not meta-commentary)
                if len(revised_answer) > 50 and not revised_answer.lower().startswith(("the answer should", "to accurately", "upon reviewing")):
                    final = revised_answer
                    improvement_note = f"✅ Review improved answer (confidence: {confidence_score}/100): {critique[:80]}"
                else:
                    # Revised answer is meta-commentary, keep original
                    logger.warning(f"⚠️ Review provided meta-commentary instead of complete answer, keeping original")
                    final = state.final_answer
                    improvement_note = f"⚠️ Review feedback (confidence: {confidence_score}/100): {critique[:80]}"
            else:
                final = state.final_answer
                improvement_note = f"✅ Review approved (confidence: {confidence_score}/100): {critique[:80]}"
            
            # Add quality metadata to reasoning
            quality_report = f"📊 Quality: {confidence_score}/100 | Sources: {num_sources} | Accuracy: {'✓' if accuracy_issues == 'NONE' else '⚠️'}"
            
            return {
                **state.model_dump(),
                "final_answer": final,
                "confidence_score": confidence_score,
                "is_complete": True,
                "reasoning": state.reasoning + [improvement_note, quality_report]
            }
            
        except Exception as e:
            logger.error(f"Review agent failed: {e}", exc_info=True)
            # On error, approve original with warning
            return {
                **state.model_dump(),
                "is_complete": True,
                "reasoning": state.reasoning + [f"⚠️ Review failed: {str(e)}, using original answer"]
            }

    @graph.add_node
    def get_total_pages_node(state: AgentState):
        # Get filename from doc_ids or key_entities
        filename = None
        if state.doc_ids:
            filename = state.doc_ids[0]
        elif state.key_entities:
            filename = state.key_entities[0]
        
        if not filename:
            # Return error if no document specified
            return {
                **state.model_dump(),
                "reasoning": state.reasoning + ["⚠️ No document specified for get_total_pages"],
            }
        
        result = agent_tools.get_total_pages(filename)
        payload = json.dumps(result)[:2000]
        chunk = {
            "content": f"TOTAL_PAGES: {payload}",
            "filename": filename,
            "page": None,
            "doc_id": "totals",
        }
        return {
            **state.model_dump(),
            "context": state.context + [chunk],
            "tools_used": state.tools_used + ["get_total_pages"],
            "reasoning": state.reasoning + ["Used get_total_pages"],
        }

    @graph.add_node
    def extract_table_of_contents_node(state: AgentState):
        doc_identifier = (
            state.doc_ids[0]
            if state.doc_ids
            else (state.key_entities[0] if state.key_entities else "")
        )
        result = (
            agent_tools.extract_table_of_contents(doc_identifier)
            if doc_identifier
            else {"toc": []}
        )
        display_name = doc_identifier if doc_identifier else "Unknown document"
        payload = json.dumps(result)[:2000]
        chunk = {
            "content": f"TOC: {payload}",
            "filename": display_name,
            "page": None,
            "doc_id": "toc",
        }
        return {
            **state.model_dump(),
            "context": state.context + [chunk],
            "tools_used": state.tools_used + ["extract_table_of_contents"],
            "reasoning": state.reasoning + ["Used extract_table_of_contents"],
        }

    @graph.add_node
    def find_figures_and_tables_node(state: AgentState):
        doc_identifier = (
            state.doc_ids[0]
            if state.doc_ids
            else (state.key_entities[0] if state.key_entities else "")
        )
        result = (
            agent_tools.find_figures_and_tables(doc_identifier)
            if doc_identifier
            else {"figures_tables": []}
        )
        display_name = doc_identifier if doc_identifier else "Unknown document"
        payload = json.dumps(result)[:2000]
        chunk = {
            "content": f"FIGURES_TABLES: {payload}",
            "filename": display_name,
            "page": None,
            "doc_id": "figtab",
        }
        return {
            **state.model_dump(),
            "context": state.context + [chunk],
            "tools_used": state.tools_used + ["find_figures_and_tables"],
            "reasoning": state.reasoning + ["Used find_figures_and_tables"],
        }

    @graph.add_node
    def extract_references_node(state: AgentState):
        doc_identifier = (
            state.doc_ids[0]
            if state.doc_ids
            else (state.key_entities[0] if state.key_entities else "")
        )
        result = (
            agent_tools.extract_references(doc_identifier)
            if doc_identifier
            else {"references": []}
        )
        display_name = doc_identifier if doc_identifier else "Unknown document"
        payload = json.dumps(result)[:2000]
        chunk = {
            "content": f"REFERENCES: {payload}",
            "filename": display_name,
            "page": None,
            "doc_id": "refs",
        }
        return {
            **state.model_dump(),
            "context": state.context + [chunk],
            "tools_used": state.tools_used + ["extract_references"],
            "reasoning": state.reasoning + ["Used extract_references"],
        }

    @graph.add_node
    def extract_emails_node(state: AgentState):
        doc_identifier = (
            state.doc_ids[0]
            if state.doc_ids
            else (state.key_entities[0] if state.key_entities else "")
        )
        result = (
            agent_tools.extract_emails(doc_identifier)
            if doc_identifier
            else {"emails": []}
        )
        display_name = doc_identifier if doc_identifier else "Unknown document"
        payload = json.dumps(result)[:2000]
        chunk = {
            "content": f"EMAILS: {payload}",
            "filename": display_name,
            "page": None,
            "doc_id": "emails",
        }
        return {
            **state.model_dump(),
            "context": state.context + [chunk],
            "tools_used": state.tools_used + ["extract_emails"],
            "reasoning": state.reasoning + ["Used extract_emails"],
        }

    @graph.add_node
    def list_all_chunks_metadata_node(state: AgentState):
        """List metadata for all chunks, optionally filtered by filename."""
        # Get filename filter from doc_ids or key_entities if specified
        filename = None
        if state.doc_ids:
            filename = state.doc_ids[0]
        elif state.key_entities:
            filename = state.key_entities[0]
        
        try:
            result = agent_tools.list_all_chunks_metadata(filename=filename)
            
            # Format the metadata list for context
            if result:
                # Limit to first 50 chunks to avoid overwhelming context
                chunks_preview = result[:50]
                summary = f"Found {len(result)} total chunks"
                if filename:
                    summary += f" for document '{filename}'"
                
                # Create a readable summary
                metadata_summary = f"{summary}\n\nSample chunks:\n"
                for i, meta in enumerate(chunks_preview[:10], 1):
                    metadata_summary += f"{i}. {meta.get('filename', 'N/A')} - Page {meta.get('page_number', 'N/A')} - Chunk ID: {meta.get('chunk_id', 'N/A')}\n"
                
                if len(result) > 10:
                    metadata_summary += f"\n... and {len(result) - 10} more chunks"
                
                chunk = {
                    "content": metadata_summary,
                    "source": "list_all_chunks_metadata",
                    "metadata": result  # Store full metadata for potential use
                }
            else:
                chunk = {
                    "content": "No chunks found" + (f" for document '{filename}'" if filename else ""),
                    "source": "list_all_chunks_metadata"
                }
            
            return {
                **state.model_dump(),
                "context": state.context + [chunk],
                "tools_used": state.tools_used + ["list_all_chunks_metadata"],
                "reasoning": state.reasoning + [f"Listed chunk metadata{' for ' + filename if filename else ''}: {len(result)} chunks found"],
            }
        except Exception as e:
            logger.error(f"list_all_chunks_metadata_node failed: {e}")
            return {
                **state.model_dump(),
                "reasoning": state.reasoning + [f"Error listing chunk metadata: {str(e)}"],
            }

    @graph.add_node
    def get_chunk_by_metadata_node(state: AgentState):
        """Retrieve a specific chunk by filename and chunk_id."""
        # Extract filename and chunk_id from state
        filename = None
        chunk_id = None
        
        # Try to get filename from doc_ids or key_entities
        if state.doc_ids:
            filename = state.doc_ids[0]
        elif state.key_entities:
            filename = state.key_entities[0]
        
        # Try to extract chunk_id from query or refined_query
        query_text = state.refined_query or state.query
        
        # Look for patterns like "chunk_id: X", "chunk X", "id X", etc.
        chunk_id_patterns = [
            r"chunk[_\s]id[:\s]+(\S+)",
            r"chunk[:\s]+(\d+)",
            r"id[:\s]+(\S+)",
        ]
        
        for pattern in chunk_id_patterns:
            match = re.search(pattern, query_text.lower())
            if match:
                chunk_id = match.group(1)
                break
        
        if not filename or not chunk_id:
            error_msg = f"Missing required parameters: filename={filename}, chunk_id={chunk_id}"
            logger.warning(error_msg)
            return {
                **state.model_dump(),
                "reasoning": state.reasoning + [f"❌ {error_msg}"],
                "tools_used": state.tools_used + ["get_chunk_by_metadata"],
            }
        
        try:
            result = agent_tools.get_chunk_by_metadata(filename=filename, chunk_id=chunk_id)
            
            if "error" in result:
                chunk = {
                    "content": result["error"],
                    "source": "get_chunk_by_metadata"
                }
                reasoning_msg = f"Chunk not found: {filename}, chunk_id={chunk_id}"
            else:
                chunk = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "source": "get_chunk_by_metadata",
                    "filename": result["metadata"].get("filename"),
                    "page": result["metadata"].get("page_number")
                }
                reasoning_msg = f"Retrieved chunk {chunk_id} from {filename} (page {result['metadata'].get('page_number', 'N/A')})"
            
            return {
                **state.model_dump(),
                "context": state.context + [chunk],
                "tools_used": state.tools_used + ["get_chunk_by_metadata"],
                "reasoning": state.reasoning + [reasoning_msg],
            }
        except Exception as e:
            logger.error(f"get_chunk_by_metadata_node failed: {e}")
            return {
                **state.model_dump(),
                "reasoning": state.reasoning + [f"Error retrieving chunk: {str(e)}"],
            }

    def route_based_on_tool(state: AgentState):
        return state.next_tool

    def route_based_on_completion(state: AgentState):
        return END if state.is_complete else "select_tool"

    def route_after_analyze(state: AgentState) -> str:
        """Always route to select_tool - let ReAct agent decide everything."""
        return "select_tool"

    # Simplified routing: analyze -> select_tool (ReAct decides from there)
    graph.add_edge("analyze_query", "select_tool")
    # Route from select_tool to all possible tool nodes
    graph.add_conditional_edges(
        "select_tool",
        route_based_on_tool,
        {
            "summarize_document": "summarize_document",
            "search_specific_documents": "search_specific_documents",
            "search_specific_page": "search_specific_page",
            "broad_search_all_documents": "broad_search_all_documents",
            "list_all_documents": "list_documents",
            "get_document_overview": "get_document_overview",
            "select_document": "select_document",
            "expand_query": "expand_query",
            "get_total_pages": "get_total_pages_node",
            "extract_table_of_contents": "extract_table_of_contents_node",
            "find_figures_and_tables": "find_figures_and_tables_node",
            "extract_references": "extract_references_node",
            "extract_emails": "extract_emails_node",
            "list_all_chunks_metadata": "list_all_chunks_metadata_node",
            "get_chunk_by_metadata": "get_chunk_by_metadata_node",
            "synthesize_answer": "synthesize_answer",
        },
    )
    
    # All data-gathering tools loop back to select_tool (ReAct continues)
    graph.add_edge("get_total_pages_node", "select_tool")
    graph.add_edge("extract_table_of_contents_node", "select_tool")
    graph.add_edge("find_figures_and_tables_node", "select_tool")
    graph.add_edge("extract_references_node", "select_tool")
    graph.add_edge("extract_emails_node", "select_tool")
    graph.add_edge("list_all_chunks_metadata_node", "select_tool")
    graph.add_edge("get_chunk_by_metadata_node", "select_tool")
    graph.add_edge("select_document", "select_tool")
    graph.add_edge("summarize_document", "select_tool")
    graph.add_edge("search_specific_documents", "select_tool")
    graph.add_edge("search_specific_page", "select_tool")
    graph.add_edge("broad_search_all_documents", "select_tool")
    graph.add_edge("get_document_overview", "select_tool")
    graph.add_edge("expand_query", "select_tool")
    graph.add_edge("list_documents", "select_tool")
    
    # NEW: synthesize_answer -> review_answer -> END
    graph.add_edge("synthesize_answer", "review_answer")
    graph.add_edge("review_answer", END)
    
    graph.set_entry_point("analyze_query")
    return graph.compile(
        # Set recursion limit high enough for complex multi-step reasoning
        # This prevents "GRAPH_RECURSION_LIMIT" errors during agent execution
        checkpointer=None,  # No checkpointing for now
        debug=False,
    )
