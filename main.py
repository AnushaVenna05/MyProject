from profanity_check import predict, predict_prob

print(predict([
  'predict() takes an array and returns a 1 for each string if it is offensive, else 0.',
  'fuck you',
]))

print(predict_prob([
  'predict_prob() takes an array and returns the probability each string is offensive',
  'go to hell, you scum','This bitch is a cunt!'
]))
