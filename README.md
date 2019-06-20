# punctuator
JP puncuator, tained on BCCWJ

# Result with Japanese BCCWJ corpus

## LSTM 
```
                  precision    recall    f1-score   support
      Blank       0.96         0.99      0.97       16920414
      Comma       0.67         0.20      0.31       594182
     Period       0.68         0.42      0.52       446364
avg / total       0.94         0.95      0.94       17960960
```

## BiLSTM
```
                  precision    recall    f1-score   support
     Blank        0.98         0.99      0.98       16920414
     Comma        0.70         0.51      0.59       594182
    Period        0.81         0.75      0.78       446364
avg / total       0.96         0.97      0.97       17960960
```

## CBiLSTM
```
                  precision    recall    f1-score   support
      Blank       0.98         0.99      0.98       16920414
      Comma       0.70         0.53      0.60       594182
     Period       0.82         0.76      0.79       446364
avg / total       0.97         0.97      0.97       17960960
```

## CLSTM
```
                  precision    recall    f1-score   support
      Blank       0.97         0.99      0.98       16920414
      Comma       0.69         0.43      0.53       594182
     Period       0.79         0.69      0.73       446364
avg / total       0.96         0.96      0.96       17960960
```

## Summary for comma and period
```
Model   C-Precision C-Recall    P-Precision P-Recall
LSTM    0.67        0.20        0.68        0.42
CLSTM   0.69        0.43        0.79        0.69
BiLSTM  0.70        0.51        0.81        0.75
CBiLSTM 0.70        0.53        0.82        0.76
```
