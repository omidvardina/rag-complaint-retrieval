import pandas as pd

comparison_df = pd.DataFrame({

    "method": [
        "fixed_size",
        "recursive",
        "token_aware_semantic"
    ],

    # Previous retrieval metrics
    "exact_id_recall_at_5": [
        0.17,
        0.13,
        0.11
    ],

    "product_match_at_5": [
        0.72,
        0.73,
        0.70
    ],

    "issue_match_at_5": [
        0.49,
        0.50,
        0.49
    ],

    "company_match_at_5": [
        0.39,
        0.38,
        0.29
    ],

    "product_issue_match_at_5": [
        0.46,
        0.47,
        0.46
    ],

    # LLM-as-judge primary evaluation
    "llm_primary_score": [
        0.725,
        0.720,
        0.670
    ],

    # Cross-encoder evaluation
    "cross_encoder_avg_score": [
        3.6187,
        3.6422,
        2.8137
    ]
})

print(comparison_df)

comparison_df.to_csv(
    "final_chunking_method_comparison.csv",
    index=False
)

print("\nSaved to final_chunking_method_comparison_100.csv")