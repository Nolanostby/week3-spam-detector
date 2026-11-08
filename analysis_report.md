Email Spam Detector Analysis

Dataset and features

My dataset has 16 original emails: 8 spam and 8 legitimate. I made two slightly changed copies of each one, so there are 48 rows total. The starter repository did not include the email template mentioned in the assignment, so I wrote my own examples. The prepare_data.py script creates the same dataset each time it runs.

The model does not read the email itself. It only uses six numbers: word_count, exclamations, money_words, all_caps, link_count, and urgency_words. The first four were part of the assignment. I added two features:

link_count counts links in an email. Spam sometimes uses links to send people to a fake prize or account page, so I thought this could help identify it.

urgency_words counts words like “urgent,” “today,” and “now.” Spam often tries to pressure people into acting quickly, so I thought this might be a useful clue.

These features are not always reliable. A real email can have a link or need a quick reply, too.

Training and results

I split the emails into training and test sets. The training set is what the model learns from, and the test set checks how it does on emails it did not train on. If I tested it using the same emails it learned from, I would mostly be checking whether it remembered them.

I used 36 emails for training and 12 for testing. The model got 100% accuracy on the test set. Here is the confusion matrix:

Actual legitimate: 6 predicted legitimate, 0 predicted spam
Actual spam:       0 predicted legitimate, 6 predicted spam

The model’s spam precision was 100%. That means every email it marked as spam in the test set really was spam. Its spam recall was also 100%. That means it caught every spam email in the test set.

Even though those results look great, there is a major problem with this test, explained below. The score does not mean the model would do this well on real email.

Which mistake is worse?

I think it is worse to send a real email to the spam folder than to let some spam into the inbox. Someone might miss an important message, like a job offer, and never realize it was there. I would make the filter less likely to label an email as spam and send unsure emails to a review folder instead. That could let more spam through, but it would also help protect important messages.

What the model learned

The model’s strongest feature was money_words, with a weight of 1.203. After that came urgency_words at 1.083, exclamations at 1.055, and link_count at 0.713. Word_count had a weight of -0.172, and all_caps had a weight of 0.109.

A positive weight pushes the model more toward predicting spam. A negative weight pushes it more toward predicting legitimate. The results mostly matched what I expected: money, urgency, exclamation marks, and links were useful clues. In this small dataset, longer emails leaned a little more toward legitimate.

Trying to fool the model

I tested the model with three extra emails:

1. A plain-sounding spam email about a job and an offer. The model called it legitimate, so this one fooled the filter.
2. A real-sounding team message that used words like “urgent,” “deadline,” and “now.” The model called it spam, so this one was a false alarm.
3. A normal message with a link to a course schedule. The model called it legitimate, which was correct.

These are only three examples, so they do not prove how the model would handle every kind of spam.

Near-copies and confidence in the score

Every original email has two lightly edited copies. All 8 email sources that appeared in the test set also had copies in the training set. This means the model had already seen very similar emails before taking its test.

Because of that, the 100% accuracy is probably too high. I would not trust it as a measure of how well the model works on new emails. A better test would make sure that all copies of an original email stay together in either training or testing, never both. Even then, 16 original emails are not enough to prove the model is good.

What I would need before trusting this filter

I would need many more emails from different people, including emails collected at different times. I would also want to test the model without putting copies of the same original email in both groups. Then I would check the mistakes it makes and adjust it based on which mistakes are more harmful. Spam changes over time, so I would keep checking its results. This project is a small experiment, not a filter I would use on real email yet.
