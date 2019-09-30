import argparse
import json
import os
from collections import defaultdict
import numpy as np
import math

def softmax(scores):
  max = np.max(scores)
  stable_x = np.exp(scores - max)
  prob = stable_x / np.sum(stable_x)
  return prob

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-1.0 * x))

def print_action(rgb_score, flow_score):
  rgb_score = sigmoid(np.array(rgb_score))
  flow_score = sigmoid(np.array(flow_score))
  fused_score = (rgb_score + 2.0 * flow_score) / 3.0
  print("RGB action {0} with probability {1}".format(np.argmax(rgb_score), np.max(rgb_score)))
  print("Flow action {0} with probability {1}".format(np.argmax(flow_score), np.max(flow_score)))
  print("Fused action {0} with probability {1}".format(np.argmax(fused_score), np.max(fused_score)))

parser = argparse.ArgumentParser()
parser.add_argument('rgb_dir', type=str)
parser.add_argument('flow_dir', type=str)
parser.add_argument('save_dir', type=str)

args = parser.parse_args()
rgb_dir = args.rgb_dir
flow_dir = args.flow_dir
save_dir = args.save_dir

filelist = os.listdir(rgb_dir)
for file in filelist:
  rgb_file = os.path.join(rgb_dir, file)
  flow_file = os.path.join(flow_dir, file)
  if not os.path.exists(flow_file):
    continue

  try:
    with open(rgb_file) as json_data:
      rgb_scores = json.load(json_data)

    with open(flow_file) as json_data:
      flow_scores = json.load(json_data)
  except:
    continue

  print("File: {0} rgb_scores {1} flow_scores {2}".format(file, len(rgb_scores), len(flow_scores)))
  output_dict = defaultdict(lambda: defaultdict(list)) 
  for key in rgb_scores:
    if key in flow_scores:
      # key might not exist in the flow because of boundary conditions.
      output_dict[key]["rgb_scores"] = rgb_scores[key]["scores"]
      output_dict[key]["flow_scores"] = flow_scores[key]["scores"]
      print_action(rgb_scores[key]["scores"], flow_scores[key]["scores"])
    else:
      print("Key {0} does not exist in flow scores".format(key))

  with open(save_dir + "/" + file, 'w') as outfile:
    json.dump(output_dict, outfile)
