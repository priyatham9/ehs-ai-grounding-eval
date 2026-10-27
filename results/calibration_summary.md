| adapter | accuracy | mean confidence | Brier [95% CI] | Brier, constant at accuracy | ECE (10 bins) [95% CI] |
|---|---|---|---|---|---|
| random_floor | 52.1% | n/a | n/a (no confidence reported) | n/a | n/a |
| retrieval_tfidf | 27.0% | 0.447 | 0.202 [0.182, 0.221] | 0.197 | 0.192 [0.128, 0.307] |
| retrieval_bm25 | 28.6% | n/a | n/a (not on a [0, 1] scale) | n/a | n/a |
| retrieval_bm25plus | 27.0% | n/a | n/a (not on a [0, 1] scale) | n/a | n/a |
| retrieval_rrf | 27.0% | n/a | n/a (not on a [0, 1] scale) | n/a | n/a |
| oracle | 100.0% | n/a | n/a (no confidence reported) | n/a | n/a |

Reliability bins, retrieval_tfidf:

| bin | items | mean confidence | accuracy |
|---|---|---|---|
| [0.2, 0.3) | 8 | 0.272 | 12.5% |
| [0.3, 0.4) | 15 | 0.364 | 0.0% |
| [0.4, 0.5) | 21 | 0.452 | 33.3% |
| [0.5, 0.6) | 14 | 0.540 | 42.9% |
| [0.6, 0.7) | 3 | 0.657 | 33.3% |
| [0.7, 0.8) | 2 | 0.766 | 100.0% |
