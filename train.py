import argparse
import gc
from agent.agent import Agent
from functions import *
import tensorflow as tf  # Import TensorFlow

def main(stock_name, window_size, episode_count, device):
    agent = Agent(window_size)

    batch_size = 32
    data = getStockDataVec(stock_name)
    l = len(data) - 1

    for e in range(episode_count + 1):
        print("Starting episode", e)
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []
        memory_batch = []

        for t in range(l):
            action = agent.act(state)

            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:
                agent.inventory.append(data[t])

            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price

            done = True if t == l - 1 else False
            memory_batch.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")

            if len(memory_batch) >= batch_size:
                agent.memory.extend(memory_batch)
                agent.expReplay(batch_size)
                memory_batch = []

        print("Value of e:", e)
        if e % 1 == 0:
            model_path = "models/model_ep" + str(e)
            print("Saving model at:", model_path)
            try:
                agent.model.save(model_path)
                print("Model saved successfully at:", model_path)
            except Exception as e:
                print("Error saving model:", str(e))
        
        # Explicitly clean up memory
        del memory_batch
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock trading training")
    parser.add_argument("stock", type=str, help="Stock name")
    parser.add_argument("window", type=int, help="Window size")
    parser.add_argument("episodes", type=int, help="Number of episodes")
    args = parser.parse_args()
    
    # Determine the device (CPU or GPU)
    device = '/device:GPU:0' if tf.test.is_gpu_available() else '/device:CPU:0'
    main(args.stock, args.window, args.episodes, device)
