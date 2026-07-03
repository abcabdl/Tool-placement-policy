# Representation Signal Report

This diagnostic asks where tool knowledge should live before training a placement policy.

## Headline

- Tools analyzed: 1032
- Tools with observed calls: 82
- Catalog-only tools without observed calls: 950
- Average full card chars: 872.6
- Average compact card chars: 339.3
- Average compact/full ratio: 0.452
- High context/retrieval signal among observed tools: 0
- High vocab/template signal among observed tools: 81
- High parameter/adapter signal among observed tools: 51
- Observed server counts: {'Bright Data': 60, 'Google News and Trends': 5, 'Pokémon': 4, 'Weather MCP Server': 3, 'Lotus Wisdom': 2, 'Weather Query Server': 2, 'Human Messages Prompt Server': 1, 'Calculator': 1, '12306 MCP Server': 1, 'Weather360 Server': 1, 'AAAAAA MCP Server': 1, 'Remote Shell Server': 1}

## How To Read The Scores

- `context_retrieval_score`: stronger for long-tail tools, weaker repeated templates, compact-card retrievability, and moderate schema complexity.
- `vocab_template_score`: stronger for frequent calls with stable argument-key/type templates and reusable JSON call skeletons.
- `parameter_adapter_score`: stronger for frequent, stable, schema-heavy tools whose knowledge is costly to keep in prompts or hard to retrieve.
- `catalog_only/context_retrieval_candidate` means a schema exists but no local calls were observed; treat it as a long-tail prior, not direct evidence.

### Tools Suited To Context/Retrieval

| Tool | Server | Calls | Score | Compact R@10 | Template Share | Full Chars | Recommendation |
|---|---|---:|---:|---:|---:|---:|---|
| confluence_conf_search | mcp server atlassian confluence | 0 | 0.743 |  | 0.000 | 4338 | catalog_only/context_retrieval_candidate |
| create_feishu_code_block | feishu mcp_2 | 0 | 0.731 |  | 0.000 | 3123 | catalog_only/context_retrieval_candidate |
| delete_feishu_document_blocks | feishu mcp_2 | 0 | 0.698 |  | 0.000 | 2067 | catalog_only/context_retrieval_candidate |
| pack_remote_repository | repomix | 0 | 0.697 |  | 0.000 | 2137 | catalog_only/context_retrieval_candidate |
| create_feishu_heading_block | feishu mcp_2 | 0 | 0.696 |  | 0.000 | 2535 | catalog_only/context_retrieval_candidate |
| create_feishu_list_block | feishu mcp_2 | 0 | 0.695 |  | 0.000 | 2394 | catalog_only/context_retrieval_candidate |
| project-explorer_search_files | Project Explorer | 0 | 0.690 |  | 0.000 | 4553 | catalog_only/context_retrieval_candidate |
| pack_codebase | repomix | 0 | 0.690 |  | 0.000 | 2001 | catalog_only/context_retrieval_candidate |
| confluence_conf_ls_pages | mcp server atlassian confluence | 0 | 0.687 |  | 0.000 | 3947 | catalog_only/context_retrieval_candidate |
| batch_create_feishu_blocks | feishu mcp_2 | 0 | 0.680 |  | 0.000 | 13038 | catalog_only/context_retrieval_candidate |
| get_notes | productboard mcp | 0 | 0.677 |  | 0.000 | 2391 | catalog_only/context_retrieval_candidate |
| create_feishu_text_block | feishu mcp_2 | 0 | 0.675 |  | 0.000 | 4625 | catalog_only/context_retrieval_candidate |

### Tools With Vocab/Template Compression Signal

| Tool | Server | Calls | Score | Compact R@10 | Template Share | Full Chars | Recommendation |
|---|---|---:|---:|---:|---:|---:|---|
| brightdata-mcp_extract | Bright Data | 30 | 1.000 | 0.400 | 1.000 | 765 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_linkedin_people_search | Bright Data | 30 | 0.982 | 0.100 | 1.000 | 628 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_google_maps_reviews | Bright Data | 30 | 0.976 | 0.800 | 1.000 | 588 | vocab/template;parameter/adapter |
| brightdata-mcp_search_engine | Bright Data | 30 | 0.968 | 0.133 | 1.000 | 704 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_amazon_product_search | Bright Data | 28 | 0.967 | 0.821 | 1.000 | 688 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_reuter_news | Bright Data | 30 | 0.960 | 0.567 | 1.000 | 499 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_youtube_comments | Bright Data | 29 | 0.960 | 0.621 | 1.000 | 590 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_type | Bright Data | 30 | 0.945 | 0.867 | 1.000 | 661 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_linkedin_company_profile | Bright Data | 28 | 0.944 | 0.786 | 1.000 | 482 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_navigate | Bright Data | 30 | 0.943 | 0.333 | 1.000 | 423 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_click | Bright Data | 30 | 0.941 | 0.633 | 1.000 | 514 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_yahoo_finance_business | Bright Data | 28 | 0.937 | 0.643 | 1.000 | 525 | vocab/template;parameter/adapter |

### Tools Worth Considering For Parameter/Adapter Placement

| Tool | Server | Calls | Score | Compact R@10 | Template Share | Full Chars | Recommendation |
|---|---|---:|---:|---:|---:|---:|---|
| brightdata-mcp_search_engine | Bright Data | 30 | 0.774 | 0.133 | 1.000 | 704 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_linkedin_people_search | Bright Data | 30 | 0.768 | 0.100 | 1.000 | 628 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_screenshot | Bright Data | 29 | 0.749 | 0.000 | 1.000 | 502 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_scroll | Bright Data | 30 | 0.734 | 0.067 | 1.000 | 274 | vocab/template;parameter/adapter |
| brightdata-mcp_extract | Bright Data | 30 | 0.728 | 0.400 | 1.000 | 765 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_navigate | Bright Data | 30 | 0.725 | 0.333 | 1.000 | 423 | vocab/template;parameter/adapter |
| brightdata-mcp_session_stats | Bright Data | 30 | 0.719 | 0.200 | 1.000 | 278 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_get_text | Bright Data | 28 | 0.716 | 0.000 | 1.000 | 276 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_links | Bright Data | 30 | 0.705 | 0.333 | 1.000 | 383 | vocab/template;parameter/adapter |
| brightdata-mcp_web_data_reuter_news | Bright Data | 30 | 0.700 | 0.567 | 1.000 | 499 | vocab/template;parameter/adapter |
| brightdata-mcp_scrape_as_markdown | Bright Data | 30 | 0.693 | 0.633 | 1.000 | 538 | vocab/template;parameter/adapter |
| brightdata-mcp_scraping_browser_click | Bright Data | 30 | 0.692 | 0.633 | 1.000 | 514 | vocab/template;parameter/adapter |

## Next Experimental Move

1. Keep compact/retrieval as the low-cost long-tail baseline.
2. Add a template-macro baseline for high `vocab_template_score` tools before attempting tokenizer changes.
3. Treat high `parameter_adapter_score` tools as candidates for later ParaTool-style modules, not as a first-week baseline.
4. Report observed-only numbers in the main text and catalog-only numbers as coverage analysis.
5. Train any policy only after these scores predict oracle winners better than frequency-only heuristics.
