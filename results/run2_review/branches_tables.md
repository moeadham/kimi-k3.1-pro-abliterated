## 1. Coverage

| set | split | original | answers | with branches | substitutions | subs/answer | wide keys | tight keys |
|---|---|---|---|---|---|---|---|---|
| borderline | harmful | answered | 180 | 100.0% (180/180) | 1801 | 10.0 | 27 | 153 |
| borderline | harmful | refused | 420 | 100.0% (420/420) | 3748 | 8.9 | 111 | 309 |
| borderline | harmless | answered | 372 | 100.0% (372/372) | 2992 | 8.0 | 0 | 372 |
| borderline | harmless | refused | 28 | 100.0% (28/28) | 144 | 5.1 | 0 | 28 |
| pairs | harmful | answered | 33 | 3.0% (1/33) | 10 | 10.0 | 0 | 1 |
| pairs | harmful | refused | 1028 | 23.2% (239/1028) | 1379 | 5.8 | 0 | 239 |
| pairs | harmless | answered | 1018 | 0.0% (0/1018) | 0 | - | 0 | 0 |
| pairs | harmless | refused | 43 | 0.0% (0/43) | 0 | - | 0 | 0 |

| version | answers | positions/answer | subs/answer | branches.jsonl lines | with control |
|---|---|---|---|---|---|
| wide | 138 | 5.59 | 16.8 | 0-2514 | 138 |
| tight | 1102 | 3.53 | 7.0 | 2393-11298 | 1087 |

## 2. Flip rates

### by original outcome x set (substitutions only; control = original token re-fed)

| original | set | substitution flip | control flip |
|---|---|---|---|
| refused | borderline | 6.0% (235/3892) | 1.6% (7/448) |
| refused | pairs | 2.8% (38/1379) | 0.0% (0/224) |
| answered | borderline | 29.0% (1391/4793) | 18.5% (102/552) |
| answered | pairs | 80.0% (8/10) | 100.0% (1/1) |

### by absolute answer token index

| answer token | refused: subs flip | refused: control flip | answered: subs flip | answered: control flip |
|---|---|---|---|---|
| 0 | 12.5% (184/1469) | 10.2% (5/49) | 44.1% (500/1133) | 29.1% (50/172) |
| 1 | 1.9% (28/1462) | 0.4% (2/559) | 37.9% (429/1133) | 0.0% (0/1) |
| 2 | 4.0% (51/1269) | 0.0% (0/1) | 32.1% (42/131) | 21.7% (5/23) |
| 3-5 | 0.0% (0/561) | - | 28.2% (42/149) | 25.0% (2/8) |
| 6-14 | 2.6% (5/189) | 0.0% (0/26) | 27.7% (39/141) | 18.2% (4/22) |
| 15-39 | 1.8% (5/277) | 0.0% (0/32) | 10.2% (127/1250) | 5.9% (12/203) |
| 40-99 | 0.0% (0/38) | 0.0% (0/4) | 23.6% (181/768) | 20.4% (22/108) |
| 100+ | 0.0% (0/6) | 0.0% (0/1) | 39.8% (39/98) | 50.0% (8/16) |

### by position relative to the Jev candidate (window positions; head-only positions excluded)

| i - candidate | refused wide | refused tight | answered wide | answered tight |
|---|---|---|---|---|
| -3 | - | - | - | 42.9% (6/14) |
| -2 | 3.0% (1/33) | 0.0% (0/2) | 23.5% (12/51) | 33.3% (16/48) |
| -1 | 12.4% (39/314) | 9.6% (101/1051) | 23.5% (12/51) | 18.0% (131/728) |
| 0 | 0.0% (0/315) | 0.3% (3/1040) | 38.3% (31/81) | 27.1% (282/1040) |
| 1 | 2.5% (8/314) | 3.9% (41/1041) | 37.0% (30/81) | 23.5% (244/1040) |
| 2 | 0.0% (0/312) | 0.0% (0/1) | 25.9% (21/81) | - |
| 3 | 0.0% (0/312) | - | 27.2% (22/81) | - |

### by alternative rank (1 = runner-up)

| rank | refused | answered | all, wide answers only |
|---|---|---|---|
| 1 | 4.7% (110/2348) | 28.9% (668/2308) | 10.4% (80/772) |
| 2 | 5.9% (139/2340) | 28.9% (666/2308) | 11.7% (90/772) |
| 3 | 4.1% (24/583) | 34.8% (65/187) | 11.6% (89/770) |

### by probability of the substituted token (p_alt) and of the original token (p_chosen)

| p_alt | refused | answered |
|---|---|---|
| <0.001 | 5.8% (13/223) | 25.3% (290/1145) |
| <0.01 | 4.5% (28/628) | 21.9% (127/579) |
| <0.05 | 3.6% (61/1693) | 29.3% (228/777) |
| <0.2 | 6.6% (140/2124) | 34.9% (588/1685) |
| <0.5 | 5.1% (31/603) | 26.9% (166/617) |
| >=0.5 | - | - |

| p_chosen | refused | answered |
|---|---|---|
| <0.3 | 15.3% (29/190) | 34.9% (146/418) |
| <0.5 | 8.6% (80/934) | 36.0% (305/848) |
| <0.7 | 6.1% (78/1280) | 30.9% (245/793) |
| <0.9 | 2.9% (43/1491) | 29.2% (218/747) |
| <0.99 | 2.4% (27/1143) | 23.5% (186/792) |
| >=0.99 | 6.9% (16/233) | 24.8% (299/1205) |

### by original-answer judge margin (flag score of the original)

| original flag score | subs flip | control flip |
|---|---|---|
| 0-0.2 | 20.9% (707/3386) | 10.4% (41/393) |
| 0.2-0.4 | 45.8% (486/1062) | 35.6% (42/118) |
| 0.4-0.6 | 58.0% (206/355) | 47.6% (20/42) |
| 0.6-0.8 | 19.7% (77/391) | 8.1% (5/62) |
| 0.8-0.95 | 8.6% (94/1097) | 1.8% (2/114) |
| 0.95-1 | 2.7% (102/3783) | 0.0% (0/496) |

## 4. Token-level patterns (head positions 0-2)

### original refused: substituted token (position 0-2), n >= 8

| i | alt token | n | flip |
|---|---|---|---|
| 0 | "I'm" | 452 | 0.0% (0/452) |
| 0 | "I'll" | 168 | 0.6% (1/168) |
| 0 | 'Creating' | 75 | 5.3% (4/75) |
| 0 | 'As' | 44 | 11.4% (5/44) |
| 0 | 'In' | 40 | 35.0% (14/40) |
| 0 | 'It' | 38 | 2.6% (1/38) |
| 0 | 'Sorry' | 35 | 0.0% (0/35) |
| 0 | '**' | 32 | 53.1% (17/32) |
| 0 | 'The' | 31 | 35.5% (11/31) |
| 0 | 'Okay' | 29 | 31.0% (9/29) |
| 0 | 'Certainly' | 29 | 37.9% (11/29) |
| 0 | 'If' | 25 | 24.0% (6/25) |
| 0 | 'I' | 24 | 0.0% (0/24) |
| 0 | 'Title' | 23 | 39.1% (9/23) |
| 0 | 'Below' | 18 | 44.4% (8/18) |
| 0 | 'To' | 14 | 7.1% (1/14) |
| 0 | 'No' | 13 | 0.0% (0/13) |
| 0 | 'Absolutely' | 13 | 30.8% (4/13) |
| 0 | 'For' | 11 | 36.4% (4/11) |
| 0 | 'Setting' | 11 | 9.1% (1/11) |
| 0 | 'This' | 10 | 0.0% (0/10) |
| 0 | 'Hyp' | 10 | 60.0% (6/10) |
| 0 | 'Developing' | 10 | 0.0% (0/10) |
| 0 | 'Here' | 8 | 87.5% (7/8) |
| 1 | " can't" | 523 | 0.0% (0/523) |
| 1 | ' cannot' | 440 | 0.0% (0/440) |
| 1 | '’m' | 160 | 0.0% (0/160) |
| 1 | ' understand' | 100 | 4.0% (4/100) |
| 1 | ' can' | 45 | 2.2% (1/45) |
| 1 | ' appreciate' | 38 | 5.3% (2/38) |
| 1 | ' need' | 27 | 3.7% (1/27) |
| 1 | " don't" | 13 | 0.0% (0/13) |
| 1 | ' should' | 12 | 16.7% (2/12) |
| 1 | ' don' | 10 | 0.0% (0/10) |
| 2 | ' help' | 463 | 5.8% (27/463) |
| 2 | ' not' | 337 | 0.0% (0/337) |
| 2 | ' provide' | 141 | 7.8% (11/141) |
| 2 | ' explain' | 65 | 3.1% (2/65) |
| 2 | ' write' | 38 | 2.6% (1/38) |
| 2 | ' certainly' | 26 | 0.0% (0/26) |
| 2 | ' assist' | 24 | 4.2% (1/24) |
| 2 | ' create' | 18 | 5.6% (1/18) |
| 2 | ' see' | 17 | 0.0% (0/17) |
| 2 | ' describe' | 12 | 8.3% (1/12) |
| 2 | ' share' | 10 | 10.0% (1/10) |
| 2 | ' discuss' | 10 | 0.0% (0/10) |

### original answered: substituted token (position 0-2), n >= 8

| i | alt token | n | flip |
|---|---|---|---|
| 0 | 'I' | 294 | 72.4% (213/294) |
| 0 | 'Certainly' | 93 | 23.7% (22/93) |
| 0 | 'Absolutely' | 74 | 17.6% (13/74) |
| 0 | "I'm" | 55 | 54.5% (30/55) |
| 0 | '**' | 37 | 43.2% (16/37) |
| 0 | 'Okay' | 28 | 25.0% (7/28) |
| 0 | 'The' | 28 | 42.9% (12/28) |
| 0 | 'As' | 27 | 55.6% (15/27) |
| 0 | 'It' | 27 | 33.3% (9/27) |
| 0 | 'Title' | 27 | 59.3% (16/27) |
| 0 | 'To' | 27 | 7.4% (2/27) |
| 0 | 'In' | 26 | 50.0% (13/26) |
| 0 | "I'll" | 22 | 63.6% (14/22) |
| 0 | 'Here' | 21 | 19.0% (4/21) |
| 0 | '"' | 19 | 31.6% (6/19) |
| 0 | '#' | 17 | 64.7% (11/17) |
| 0 | '"In' | 14 | 64.3% (9/14) |
| 0 | 'Creating' | 11 | 45.5% (5/11) |
| 0 | '###' | 11 | 72.7% (8/11) |
| 0 | "It's" | 10 | 20.0% (2/10) |
| 0 | 'Ah' | 10 | 50.0% (5/10) |
| 0 | 'If' | 9 | 55.6% (5/9) |
| 0 | 'Below' | 8 | 0.0% (0/8) |
| 0 | '*' | 8 | 25.0% (2/8) |
| 1 | '.' | 63 | 36.5% (23/63) |
| 1 | ' a' | 48 | 54.2% (26/48) |
| 1 | ',' | 43 | 25.6% (11/43) |
| 1 | '!' | 41 | 46.3% (19/41) |
| 1 | ' can' | 38 | 47.4% (18/38) |
| 1 | ' **' | 33 | 30.3% (10/33) |
| 1 | ' understand' | 28 | 50.0% (14/28) |
| 1 | ' the' | 27 | 33.3% (9/27) |
| 1 | '—' | 25 | 16.0% (4/25) |
| 1 | '’m' | 21 | 47.6% (10/21) |
| 1 | ' an' | 19 | 42.1% (8/19) |
| 1 | ' your' | 19 | 31.6% (6/19) |
| 1 | ' this' | 18 | 33.3% (6/18) |
| 1 | ' appreciate' | 17 | 47.1% (8/17) |
| 1 | ' need' | 17 | 76.5% (13/17) |
| 1 | ' "' | 15 | 46.7% (7/15) |
| 1 | " can't" | 13 | 100.0% (13/13) |
| 1 | ' hear' | 12 | 66.7% (8/12) |
| 1 | ' response' | 12 | 91.7% (11/12) |
| 1 | ' am' | 11 | 72.7% (8/11) |
| 1 | ' ' | 11 | 45.5% (5/11) |
| 1 | ' “' | 10 | 40.0% (4/10) |
| 1 | ' —' | 9 | 11.1% (1/9) |
| 1 | 'The' | 9 | 44.4% (4/9) |
| 1 | ' should' | 8 | 75.0% (6/8) |
| 1 | ' Upon' | 8 | 50.0% (4/8) |
| 1 | 'Title' | 8 | 75.0% (6/8) |
| 2 | ' Below' | 10 | 10.0% (1/10) |

### original refused: original (chosen) token at the substituted position, all positions, n >= 15

| chosen token | n | flip |
|---|---|---|
| 'I' | 1390 | 12.7% (177/1390) |
| ' can' | 1314 | 0.5% (6/1314) |
| '’t' | 1244 | 3.9% (48/1244) |
| ' help' | 193 | 0.0% (0/193) |
| ' I' | 114 | 2.6% (3/114) |
| ' with' | 103 | 0.0% (0/103) |
| ' provide' | 84 | 1.2% (1/84) |
| ' create' | 58 | 0.0% (0/58) |
| ' cannot' | 54 | 0.0% (0/54) |
| ' a' | 52 | 5.8% (3/52) |
| ',' | 30 | 16.7% (5/30) |
| ' write' | 26 | 0.0% (0/26) |
| ' won' | 26 | 0.0% (0/26) |
| ' appreciate' | 23 | 8.7% (2/23) |
| ' but' | 21 | 14.3% (3/21) |
| 'In' | 20 | 15.0% (3/20) |
| " can't" | 19 | 5.3% (1/19) |
| ' need' | 18 | 16.7% (3/18) |
| ' have' | 18 | 0.0% (0/18) |
| ' the' | 17 | 11.8% (2/17) |
| ' access' | 16 | 0.0% (0/16) |
| ' understand' | 16 | 6.2% (1/16) |

### original answered: original (chosen) token at the substituted position, all positions, n >= 15

| chosen token | n | flip |
|---|---|---|
| '.' | 613 | 14.5% (89/613) |
| ' Here' | 425 | 9.2% (39/425) |
| ' are' | 344 | 10.5% (36/344) |
| 'I' | 176 | 38.6% (68/176) |
| 'In' | 125 | 59.2% (74/125) |
| ',' | 122 | 40.2% (49/122) |
| '’s' | 102 | 11.8% (12/102) |
| '**' | 95 | 53.7% (51/95) |
| ' the' | 87 | 51.7% (45/87) |
| '.\n\n' | 83 | 33.7% (28/83) |
| 'Certainly' | 82 | 50.0% (41/82) |
| ' a' | 81 | 32.1% (26/81) |
| 'The' | 80 | 38.8% (31/80) |
| '!' | 79 | 25.3% (20/79) |
| ' can' | 75 | 69.3% (52/75) |
| 'Absolutely' | 61 | 31.1% (19/61) |
| '’m' | 54 | 46.3% (25/54) |
| 'Title' | 51 | 43.1% (22/51) |
| ' is' | 48 | 25.0% (12/48) |
| 'ing' | 45 | 17.8% (8/45) |
| ':' | 32 | 50.0% (16/32) |
| ' Below' | 29 | 13.8% (4/29) |
| ' understand' | 27 | 55.6% (15/27) |
| 'As' | 27 | 51.9% (14/27) |
| ' this' | 26 | 46.2% (12/26) |

## 5. Noise sources

- Jev repeat on the identical branch text: verdict disagrees 2.2% (249/11299).
- rejudged score within 0.4-0.8 (near the 0.6 threshold): 19.6% (2218/11299); disagreement among them 10.6% (236/2218).
- Jev repeat on the original full answer (1,240 tested answers): disagrees 0.3% (4/1240) (refused originals 0.6% (4/687), answered 0.0% (0/553)).
| original | answers with control | original longer than control text (cut) | cut original flips vs saved verdict | full original re-judged flips |
|---|---|---|---|---|
| refused | 672 | 644 | 0.8% (5/644) | 0.6% (4/644) |
| answered | 553 | 546 | 18.5% (101/546) | 0.0% (0/546) |

| original | controls | control text diverges from original | flip | identical text (judge/truncation only) | flip | diverged |
|---|---|---|---|---|
| refused | 672 | 85.0% (571/672) | 0.0% (0/101) | 1.2% (7/571) |
| answered | 553 | 65.1% (360/553) | 20.7% (40/193) | 17.5% (63/360) |

- Control divergence point: median 107 chars after the re-fed token (quartiles 56 / 191).

| original | branch | flip (saved verdict) | robust flip (saved + re-judge of branch + re-judge of original agree) |
|---|---|---|---|
| refused | substitutions | 5.2% (273/5271) | 4.9% (257/5271) |
| refused | controls | 1.0% (7/672) | 0.7% (5/672) |
| answered | substitutions | 29.1% (1399/4803) | 27.3% (1311/4803) |
| answered | controls | 18.6% (103/553) | 17.4% (96/553) |

## 7. Judge scores (re-judged) across transitions

| original | branch outcome | n | original refused / deflected (re-judge) | branch refused / deflected |
|---|---|---|---|---|
| refused | kept | 4998 | 0.92 / 0.90 | 0.90 / 0.93 |
| refused | flipped | 273 | 0.77 / 0.78 | 0.13 / 0.36 |
| answered | kept | 3404 | 0.03 / 0.14 | 0.10 / 0.29 |
| answered | flipped | 1399 | 0.06 / 0.22 | 0.43 / 0.78 |

| original | position | n | branch: refused | deflected only | borderline | answered |
|---|---|---|---|---|---|---|
| refused | i=0 | 1469 | 73.1% (1074/1469) | 14.6% (215/1469) | 4.6% (68/1469) | 7.6% (112/1469) |
| refused | i=1-2 | 2731 | 91.1% (2489/2731) | 6.0% (164/2731) | 1.7% (47/2731) | 1.1% (31/2731) |
| refused | i>=3 | 1071 | 94.4% (1011/1071) | 4.7% (50/1071) | 0.8% (9/1071) | 0.1% (1/1071) |
| answered | i=0 | 1133 | 18.0% (204/1133) | 25.4% (288/1133) | 16.4% (186/1133) | 40.2% (455/1133) |
| answered | i=1-2 | 1264 | 12.7% (160/1264) | 23.5% (297/1264) | 19.5% (246/1264) | 44.4% (561/1264) |
| answered | i>=3 | 2406 | 2.4% (57/2406) | 17.4% (418/2406) | 18.0% (433/2406) | 62.3% (1498/2406) |

Original answers (saved verdict) by category: {(True, 'refused'): 637, (False, 'answered'): 511, (False, 'borderline (0.4-0.6)'): 42, (True, 'deflected only'): 50} 

## 6b. Fork calls with clean flips

| original | tested answers | forks (saved rows) | clean forks | same position | saved fork not clean | clean fork at last tested position (censored) |
|---|---|---|---|---|---|---|
| refused | 687 | 173 | 112 | 96 | 61 | 20 |
| answered | 553 | 313 | 147 | 89 | 166 | 22 |

clean fork positions, refused: [(0, 86), (1, 3), (2, 22), (26, 1)]

clean fork positions, answered: [(0, 85), (1, 46), (2, 5), (3, 2), (5, 2), (6, 1), (10, 1), (11, 1), (42, 1), (43, 1), (59, 1), (84, 1)]

## 8. Position confound

| original | rows | n | median answer index | 0 | 1 | 2 | 3-9 | 10-49 | 50+ |
|---|---|---|---|---|---|---|---|---|---|
| refused | all rows (nt_fork anchor = at_token) | 1473 | 1 | 7.3% (107/1473) | 85.3% (1256/1473) | 3.0% (44/1473) | 0.1% (2/1473) | 4.2% (62/1473) | 0.1% (2/1473) |
| refused | confirmed forks (saved) | 173 | 0 | 61.8% (107/173) | 9.8% (17/173) | 24.9% (43/173) | 0.0% (0/173) | 3.5% (6/173) | 0.0% (0/173) |
| refused | clean forks | 112 | 0 | 76.8% (86/112) | 2.7% (3/112) | 19.6% (22/112) | 0.0% (0/112) | 0.9% (1/112) | 0.0% (0/112) |
| answered | all rows (nt_fork anchor = at_token) | 1599 | 17 | 32.0% (512/1599) | 7.2% (115/1599) | 1.4% (23/1599) | 2.8% (45/1599) | 47.0% (752/1599) | 9.5% (152/1599) |
| answered | confirmed forks (saved) | 313 | 1 | 33.2% (104/313) | 35.5% (111/313) | 1.9% (6/313) | 5.1% (16/313) | 17.3% (54/313) | 7.0% (22/313) |
| answered | clean forks | 147 | 0 | 57.8% (85/147) | 31.3% (46/147) | 3.4% (5/147) | 3.4% (5/147) | 2.7% (4/147) | 1.4% (2/147) |

prompt length median: refused 35, answered 34; fork_state = prompt_len + at_token - 1 (absolute) median refused 35, answered 57

token produced at the nt_fork anchor, refused: [(' can', 1246), ('I', 105), ('’t', 41), (' cannot', 28), (" can't", 10), (' won', 7), (' don', 3), (" don't", 3), (' but', 3), (',', 2)]

token produced at the nt_fork anchor, answered: [(' Here', 443), (' Below', 147), ('**', 82), ('Below', 64), (" Here's", 63), ('The', 56), ('.\n\n', 48), ('#', 43), ('In', 28), ('I', 20)]

## 8b. Position-matched contrast inventory

| answer index of clean fork | refused rows | answered rows |
|---|---|---|
| 0 | 86 | 85 |
| 1 | 3 | 46 |
| 2 | 22 | 5 |
| 3 | 0 | 2 |

positions 600 {('flipped', 'original', 'recovered'): 600}
| original | position | flipped branch | recovered branch | answer | n |
|---|---|---|---|---|---|
| answered | i=0 | clean flip | control as recovered |  | 15 |
| answered | i=0 | clean flip | recovered |  | 51 |
| answered | i=0 | clean flip | recovered | control-flipped answer | 8 |
| answered | i=0 | soft flip | control as recovered |  | 7 |
| answered | i=0 | soft flip | recovered |  | 82 |
| answered | i=0 | soft flip | recovered | control-flipped answer | 13 |
| answered | i=1-2 | clean flip | recovered |  | 26 |
| answered | i=1-2 | clean flip | recovered | control-flipped answer | 5 |
| answered | i=1-2 | soft flip | control as recovered |  | 1 |
| answered | i=1-2 | soft flip | recovered |  | 70 |
| answered | i=1-2 | soft flip | recovered | control-flipped answer | 22 |
| answered | i>=3 | clean flip | recovered |  | 9 |
| answered | i>=3 | clean flip | recovered | control-flipped answer | 5 |
| answered | i>=3 | soft flip | control as recovered |  | 10 |
| answered | i>=3 | soft flip | recovered |  | 76 |
| answered | i>=3 | soft flip | recovered | control-flipped answer | 36 |
| refused | i=0 | clean flip | control as recovered |  | 1 |
| refused | i=0 | clean flip | recovered |  | 58 |
| refused | i=0 | clean flip | recovered | control-flipped answer | 2 |
| refused | i=0 | soft flip | control as recovered |  | 2 |
| refused | i=0 | soft flip | recovered |  | 38 |
| refused | i=0 | soft flip | recovered | control-flipped answer | 3 |
| refused | i=1-2 | clean flip | recovered |  | 23 |
| refused | i=1-2 | clean flip | recovered | control-flipped answer | 3 |
| refused | i=1-2 | soft flip | recovered |  | 27 |
| refused | i=1-2 | soft flip | recovered | control-flipped answer | 1 |
| refused | i>=3 | soft flip | control as recovered |  | 1 |
| refused | i>=3 | soft flip | recovered |  | 5 |

positions in control-flipped answers: 98
position == rows fork_token: 328
refused/answered originals: Counter({False: 436, True: 164})
refused: n=685, tested mass median 0.90, refusal share median 1.00, mixed (both outcomes reachable at token 0) 92, share in (0.1,0.9): 62
answered: n=553, tested mass median 0.61, refusal share median 0.00, mixed (both outcomes reachable at token 0) 164, share in (0.1,0.9): 121
| original | position | branches | share of budget | clean flips | clean flips per 100 branches | share of all clean flips (this outcome) |
|---|---|---|---|---|---|---|
| refused | i=0 | 1469 | 14.6% | 112 | 7.6 | 77.8% |
| refused | i=1 | 1462 | 14.5% | 6 | 0.4 | 4.2% |
| refused | i=2 | 1269 | 12.6% | 25 | 2.0 | 17.4% |
| refused | i>=3, window | 1071 | 10.6% | 1 | 0.1 | 0.7% |
| answered | i=0 | 1133 | 11.2% | 204 | 18.0 | 49.2% |
| answered | i=1 | 1133 | 11.2% | 140 | 12.4 | 33.7% |
| answered | i=2 | 131 | 1.3% | 20 | 15.3 | 4.8% |
| answered | i>=3, window | 2406 | 23.9% | 51 | 2.1 | 12.3% |

alt-token classes (outcome, position, token) with n>=20 and zero clean flips: 37, 3544 branches (35.2% of substitutions)
[((True, 1, "can't"), 523), ((True, 0, "i'm"), 452), ((True, 1, 'cannot'), 440), ((True, 2, 'not'), 337), ((False, 3, 'is'), 196), ((True, 0, "i'll"), 168), ((True, 1, '’m'), 160), ((False, 3, '’s'), 160), ((False, 3, 'below'), 103), ((True, 2, 'explain'), 65), ((False, 3, 'and'), 61), ((True, 3, 'assist'), 57), ((False, 3, 'are'), 57), ((True, 3, 'provide'), 52), ((False, 3, 'here'), 51)]
