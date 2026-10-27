| adapter | accuracy at full coverage | AURC [95% CI] | AURC, uninformative score | AURC, best ordering | E-AURC | selective accuracy at 25% coverage | at 50% | at 75% |
|---|---|---|---|---|---|---|---|---|
| random_floor | 52.1% | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| retrieval_tfidf | 27.0% | 0.564 [0.385, 0.751] | 0.730 | 0.382 | 0.182 | 50.0% [22.2, 75.0] (answers 25%) | 40.6% [20.6, 59.4] (answers 51%) | 33.3% [18.4, 50.0] (answers 76%) |
| retrieval_bm25 | 28.6% | 0.571 [0.393, 0.765] | 0.714 | 0.362 | 0.209 | 56.2% [26.7, 81.2] (answers 25%) | 40.6% [21.9, 61.3] (answers 51%) | 31.2% [18.0, 46.0] (answers 76%) |
| retrieval_bm25plus | 27.0% | 0.602 [0.401, 0.807] | 0.730 | 0.382 | 0.219 | 43.8% [18.8, 76.5] (answers 25%) | 40.6% [20.0, 61.3] (answers 51%) | 29.2% [14.9, 44.2] (answers 76%) |
| retrieval_rrf | 27.0% | 0.699 [0.555, 0.827] | 0.730 | 0.382 | 0.316 | 30.4% [17.5, 45.0] (answers 89%) | 30.4% [17.5, 45.0] (answers 89%) | 30.4% [17.5, 45.0] (answers 89%) |
| oracle | 100.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
