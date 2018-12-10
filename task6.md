# Task 6

In general, and especially if your data is numerical, then it is a good idea to do it. You can still use non-normalized data but normalizing it makes life easier.

One common reason is feature scaling. Imagine that you trained your model on a dataset with two features `x` and `y` and your model somehow decided that the best prediction formula is `x^4 + y`. Now imagine you have the following dataset:

| x     | y   |
|-------|-----|
| 200   | 400 |
| 21000 | 100 |
| 400   | 450 |

Now if the model tries to predict the right label for those values it will compute:

```
v0 = 200^4 + 400 = 1600000400
v1 = 21000^4 + 100 = 194481000000000100
v2 = 400^4 + 450 = 25600000450
```

As you can see the computations can easily "explode" depending on the model and the scaling of the values. This is why it is so important to normalize data and to keep them in a form where explosions like this won't happen.

There are some other reasons like:
1. Consistency: we would like models and dataset to be easily comparable. So it is much better if all of them are normalized.
2. Machine learning optimizations, like the ones in the scikit learn library, work much better when the dataset is normalized.
3. Regularization behaves differently for different scaling. It will work better with normalized data.

Min max scaling consists in normalizing the data so that the minimum instance has value 0 and the maximum instance has value 1 while zero mean/unit variance normalization consists in normalizing the data so that the mean is 0 and the variance is 1. Zero mean/unit variance normalization is most commonly used but choosing one of the two really depends on the task at hand. Try going choosing zero mean/unit variance "by default" but be careful about the situation: if you have an image where pixels have values 0-255, then scaling them to 0-1 is probably better. Or, if you're using neural networks, it is likely that they work better on data that lies in the 0-1 range.
