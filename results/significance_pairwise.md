| A | B | accuracy A | accuracy B | difference A-B [95% paired cluster CI] | A only right | B only right | McNemar exact p | Holm-adjusted p | significant at 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| retrieval_tfidf | retrieval_bm25 | 27.0% | 28.6% | -1.6 pts [-4.8, +0.0] | 0 | 1 | 1.000 | 1.000 | no |
| retrieval_tfidf | retrieval_bm25plus | 27.0% | 27.0% | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 | no |
| retrieval_tfidf | retrieval_rrf | 27.0% | 27.0% | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 | no |
| retrieval_bm25 | retrieval_bm25plus | 28.6% | 27.0% | +1.6 pts [+0.0, +4.8] | 1 | 0 | 1.000 | 1.000 | no |
| retrieval_bm25 | retrieval_rrf | 28.6% | 27.0% | +1.6 pts [+0.0, +4.8] | 1 | 0 | 1.000 | 1.000 | no |
| retrieval_bm25plus | retrieval_rrf | 27.0% | 27.0% | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 | no |
| retrieval_tfidf | random_floor | 27.0% | 55.6% | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 | yes |
| retrieval_bm25 | random_floor | 28.6% | 55.6% | -27.0 pts [-46.0, -8.2] | 8 | 25 | 0.005 | 0.032 | yes |
| retrieval_bm25plus | random_floor | 27.0% | 55.6% | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 | yes |
| retrieval_rrf | random_floor | 27.0% | 55.6% | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 | yes |
