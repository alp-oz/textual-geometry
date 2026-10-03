TEXT:  Thou art more lovely and more temperate
Qwen tokens  : [' Thou', ' art', ' more', ' lovely', ' and', ' more', ' temper', 'ate']
XLM-R tokens : ['▁Tho', 'u', '▁art', '▁more', '▁lovely', '▁and', '▁more', '▁temper', 'ate']
1) THE SAME WORD AT TWO POSITIONS (positions found by searching for the token)
   Qwen contextual      distance between the two 'more' vectors:     214.71   (typical distance between neighbouring words: 190.84)
   XLM-R contextual     distance between the two 'more' vectors:       1.66   (typical distance between neighbouring words: 3.61)
   Qwen first-token     distance between the two 'more' vectors:       0.00   (typical distance between neighbouring words: 176.16)
   XLM-R first-token    distance between the two 'more' vectors:       0.00   (typical distance between neighbouring words: 3.15)
   Qwen table           distance between the two 'more' vectors:       0.00   (typical distance between neighbouring words: 0.62)
   XLM-R table          distance between the two 'more' vectors:       0.00   (typical distance between neighbouring words: 6.06)
2) FIRST THREE NUMBERS OF THE VECTOR OF ' more' (first occurrence), Qwen:
   Qwen table           [ 0.017 -0.012  0.001]  length 0.41
   Qwen first-token     [ 0.454 -2.302 -1.179]  length 243.60
   Qwen contextual      [-5.188 -2.666 -2.25 ]  length 287.52
3) SAME WORDS, TWO ORDERS:
   A = ' Thou art more lovely and more temperate'
   B = ' lovely Thou temperate more art and more'
   track                     E1 (Steele: total MST length)      path length (order-aware)
   Qwen contextual      A   1090.30  B    910.18   different   A   1335.87  B   1100.11
   XLM-R contextual     A     22.97  B     20.32   different   A     28.86  B     25.64
   Qwen first-token     A   1151.04  B   1151.04        same   A   1233.11  B   1375.33
   XLM-R first-token    A     18.88  B     18.88        same   A     25.16  B     27.40
   Qwen table           A      4.07  B      4.07        same   A      4.32  B      4.37
   XLM-R table          A     43.14  B     43.14        same   A     48.46  B     49.56
