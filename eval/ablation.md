Split: test. Queries per slice: all 149 (p@10 on 106), no brand 106, no brand, no fragrance 93, brand 43 (p@10 on 0), title overlap < 0.3 38 (p@10 on 33), text overlap < 0.7 43 (p@10 on 31).

| metric | bm25 |
|---|---|
| p@10 no brand | 0.643 [0.587, 0.699] |
| p@10 no brand, no fragrance | 0.620 [0.558, 0.681] |
| nDCG@10 no brand | 0.655 [0.612, 0.699] |
| nDCG@10 brand | 0.857 [0.789, 0.919] |
| source@10 all | 0.812 [0.745, 0.872] |
| p@10 title overlap < 0.3 | 0.621 [0.527, 0.718] |
| source@10 title overlap < 0.3 | 0.632 [0.474, 0.789] |
| p@10 text overlap < 0.7 | 0.642 [0.526, 0.752] |
| source@10 text overlap < 0.7 | 0.488 [0.349, 0.628] |
| median ms | 9.36 |
