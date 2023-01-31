import gym

import numpy as np

import statistics 

#--------------------------------------------------------------#

#Training the agent

#Initializing conditions

gamma = 0.9

rewards_array = np.array([])

epsilon_array = np.array([])

var = np.array([])

epsilon = 0.10

while epsilon <= 0.90:
         
    #Initializing Q-table 
    #It was given that there were 500 states and 6 possible actions

    V = np.zeros([500,6])

    env = gym.make("Taxi-v3").env
    
    rewards = np.array([])

    for i in range(1000): 
            
        cumulative_reward = 0
       
        done = False 
    
        state = env.reset()
    
        #Implementing epsilon greedy 
    
        while done == False: 
        
            p = np.random.rand(1,1)[0]
        
            if p < epsilon: 
            
                action = env.action_space.sample()  #Exploration 
            
            else: 
            
                action = np.argmax(V[state]) #Exploitation 
        
            observation, reward, done, info = env.step(action)
            
            cumulative_reward += reward 
            
            if done == True: 
            
                rewards = np.append(rewards, cumulative_reward)
                
            #Applying the quality function formula 
                
            V[state,action] = reward + gamma*(np.max(V[observation]))
        
            state = observation
            
    epsilon_array = np.append(epsilon_array,epsilon)
        
    rewards_array = np.append(rewards_array,np.mean(rewards))
    
    var = np.append(var,statistics.variance(rewards))
    
    print(statistics.variance(rewards))
    
    print(np.mean(rewards))
    
    epsilon += 0.1
    
    #Testing the agent 
    
    done_counter = 0
    
    for i in range(200): 
        
        done = False
        
        state = env.reset()
        
        steps = 0
        
        while done == False: 
            
            action = np.argmax(V[state])
            
            steps += 1
            
            observation, reward, done, info = env.step(action)
            
            state = observation
            
            if steps > 20:
                
                break
            
            if done == True: 
                
                done_counter += 1
                
    print(done_counter)



            
