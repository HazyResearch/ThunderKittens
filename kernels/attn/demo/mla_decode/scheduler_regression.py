from scheduler_v2 import backward_schedule
import random
from tqdm import tqdm
import time
import numpy as np
from sklearn.linear_model import LinearRegression

def get_length(seq_length, num_processors, num_tokens):
    schedule, partial_uid, reduction_uid = backward_schedule(list(range(num_processors)), 0, seq_length, list(range(num_tokens)), 0, num_processors)
    return max([t.finish for t in schedule])

def generate_random_workloads(num_workloads: int):
    for _ in range(num_workloads):
        schedule_length = 9999
        while schedule_length > 100 and random.random() > np.exp(-(schedule_length-100)/100):
            num_processors = random.randint(1, 132)
            num_tokens = random.randint(1, 4) 
            seq_length = random.randint(1, 65536)
            schedule_length = get_length(seq_length, num_processors, num_tokens)
        yield num_processors, num_tokens, seq_length, schedule_length

def make_features(num_processors, num_tokens, seq_length):
    steps = (seq_length+31)//32
    return [
        num_processors,
        num_tokens,
        steps,

        steps/num_processors,  # Work per processor
        steps//num_processors,  # Floor work per processor
        (steps+num_processors-1)//num_processors,  # Max work per processor
        steps % num_processors, # Remainder wave leftover
        num_tokens/num_processors,  # Token overhead per processor

        np.log2(steps),
        np.log2(num_processors),    # Depth of reduction tree
        num_tokens*np.log2(num_processors),    # Depth of reduction tree
        np.ceil(np.log2(num_processors)),    # Depth of reduction tree
        num_tokens*np.ceil(np.log2(num_processors)),    # Depth of reduction tree
        
        steps * np.log2(num_processors),  # Communication overhead scaling
        (steps / num_processors) * np.log2(num_processors),  # Communication overhead scaling
        
        num_processors * steps # Overall bigness
    ]

def train_model():
    NUM_WORKLOADS = 20000
    xtrain, ytrain = [], []
    for num_processors, num_tokens, seq_length, schedule_length in tqdm(generate_random_workloads(NUM_WORKLOADS), total=NUM_WORKLOADS):
        xtrain.append(make_features(num_processors, num_tokens, seq_length))
        ytrain.append(schedule_length)

    xtest, ytest = [], []
    for num_processors, num_tokens, seq_length, schedule_length in tqdm(list(generate_random_workloads(1000))):
        xtest.append(make_features(num_processors, num_tokens, seq_length))
        ytest.append(schedule_length)

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)

    model = LinearRegression()
    model.fit(xtrain, ytrain)

    return model.coef_, model.intercept_

def estimate_schedule_length(num_processors, num_tokens, seq_length):
    features = np.array(make_features(num_processors, num_tokens, seq_length))
    weights = np.array([ 8.46746941e-02,  1.68300124e+00, -5.12640395e-01,  4.99387510e-01,
                        -6.28183426e-01,  6.33647090e-01, -8.18835727e-04, -2.80419818e+00,
                        -7.89583783e-01, -1.46136190e+00, -6.85688746e-02, -7.99856622e-03,
                        -1.51483520e-01,  7.94864055e-02,  1.08745108e+00, -7.59371060e-04 ])
    return np.dot(features, weights) + 23.88058885760671

if __name__ == "__main__":
    
    for num_processors, num_tokens, seq_length, schedule_length in tqdm(list(generate_random_workloads(100))):
        print(estimate_schedule_length(num_processors, num_tokens, seq_length), schedule_length)