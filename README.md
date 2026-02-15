# auto-g-nano
A complete, minimal, fully functional decoder-only LLM (nanoGPT-style) from absolute scratch in pure PyTorch. 
No HF, no pre-built modules, no wrappers—just the essentials.


~~~console
uv run python3 train.py
Model created with 10.8M params
step    0 | train loss 4.2932 | val loss 4.2972
step  250 | train loss 2.2427 | val loss 2.2972
step  500 | train loss 1.6676 | val loss 1.8356
step  750 | train loss 1.4562 | val loss 1.6704
step 1000 | train loss 1.3553 | val loss 1.5904
step 1250 | train loss 1.2845 | val loss 1.5354
step 1500 | train loss 1.2370 | val loss 1.5030
step 1750 | train loss 1.1964 | val loss 1.4912
step 2000 | train loss 1.1615 | val loss 1.4991
step 2250 | train loss 1.1283 | val loss 1.4967
step 2500 | train loss 1.0986 | val loss 1.4848
step 2750 | train loss 1.0687 | val loss 1.4891
step 3000 | train loss 1.0411 | val loss 1.4933
step 3250 | train loss 1.0093 | val loss 1.5166
step 3500 | train loss 0.9850 | val loss 1.5085
step 3750 | train loss 0.9555 | val loss 1.5213
step 4000 | train loss 0.9252 | val loss 1.5391
step 4250 | train loss 0.8997 | val loss 1.5588
step 4500 | train loss 0.8706 | val loss 1.5613
step 4750 | train loss 0.8442 | val loss 1.5871
~~~

~~~console
uv run python3 generate.py
Model created with 10.8M params


LADY GREY:
And their I know before I descend.

KING EDWARD IV:
But what prophecy they love to do thee for thee?

GLOUCESTER:
The bridge makes her majesty to be at once.

KING EDWARD IV:
Why, then, though the time I see thee so see me speak,
Yet in thee she was many more behind my near.

LADY GREY:
What then, madam?

KING EDWARD IV:
Then is it not so?

LADY GREY:
The sentence makes us patience, and what I should
thy love to thy arms and mother and mine,
Which I am as about at all my arms
As let 
~~~