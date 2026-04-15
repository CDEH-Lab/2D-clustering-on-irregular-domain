import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Function to check normality and display distribution of simulations for the first distance meeting the criteria
def check_normality_and_display_first(K_support, K_updated_simulations, K_statistic, data_info, save_path, method=1, display='off', support=40):

    store_state = 0

    if method == 1:
        
        # Check for each distance values when the observed K-function is above the IC-95% of the simulation
        for index in range(support):
            
            # The corresponding value of the distance, simulations and observed value
            selected_distance = K_support[index]
            selected_simulations = K_updated_simulations[:, index]
            observed_value = K_statistic[index]

            # Calculate parameters of the normal distribution
            mu, sigma = np.mean(selected_simulations), np.std(selected_simulations)

            # Calculate the IC-95% of these simulations
            conf_interval = np.percentile(selected_simulations, [2.5, 97.5])

            # Display the results only if the observed value is above the IC-95% superiour value (one time only)
            if observed_value > conf_interval[1]:
                
                if store_state == 0:
                    observed_value_selected = observed_value
                    observed_distance_selected = selected_distance
                    selected_mu, selected_sigma = np.mean(selected_simulations), np.std(selected_simulations)
                    selected_conf_interval = np.percentile(selected_simulations, [2.5, 97.5])
                    used_selected_simulations = K_updated_simulations[:, index]
                    
                store_state += 1
                    
                if store_state >= 3:
                
                    # Display ON
                    if display == 'on':
                        
                        # Ensure the save directory exists
                        full_save_path = os.path.join(save_path, data_info)
                        os.makedirs(full_save_path, exist_ok=True)
                        
                        # Display the histogram of the simulations
                        plt.hist(used_selected_simulations, bins=30, density=True, alpha=0.6, color='g', label='Simulations')

                        # Generated a normal distribution curve on our data
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = stats.norm.pdf(x, selected_mu, selected_sigma)
                        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

                        # Display the observed value
                        plt.axvline(observed_value_selected, color='r', linestyle='dashed', linewidth=2, label='Observed Value')

                        # Display the IC-95%
                        plt.axvline(selected_conf_interval[0], color='b', linestyle='dashed', linewidth=2, label='IC-95%')
                        plt.axvline(selected_conf_interval[1], color='b', linestyle='dashed', linewidth=2)

                        title = f"Distance {observed_distance_selected}: mu = {mu:.2f}, sigma = {sigma:.2f}"
                        plt.title(title)
                        plt.legend()
                        
                        # Save the figure
                        file_path = os.path.join(full_save_path, 'IC_distance_detection_1st_line.png')
                        plt.savefig(file_path)
                        plt.show()

                        # Print parameters of the normal distribution and IC-95%
                        print(f"Distance: {observed_distance_selected}")
                        print(f"Mu (mean) = {selected_mu}")
                        print(f"Sigma (standard deviation) = {selected_sigma}")
                        print(f"Observed value = {observed_value_selected}")
                        print(f"Confidant Interval of 95% = [{selected_conf_interval[0]}, {selected_conf_interval[1]}]")
                        
                        # Return the distance value where we first match this condition 
                        return observed_distance_selected
                    
                    # Display OFF
                    else:

                        # Return the distance value where we first match this condition 
                        return observed_distance_selected
            else:
                store_state = 0
        return None
    
    elif method == 2:
        
        # Check for each distance values when the observed K-function is above the IC-95% of the simulation
        for index in range(1,support):
            
            # The corresponding value of the distance, simulations and observed value
            selected_simulations = K_updated_simulations[:, index]
            observed_value = K_statistic[index]

            # Calculate parameters of the normal distribution
            mu, sigma = np.mean(selected_simulations), np.std(selected_simulations)

            # Calculate the IC-95% of these simulations
            conf_interval = np.percentile(selected_simulations, [2.5, 97.5])

            # Display the results only if the observed value is above the IC-95% superiour value (one time only)
            if observed_value > conf_interval[1]:
                store_state += 1
                
                next_conf_interval = np.percentile(K_updated_simulations[:, index + 1], [2.5, 97.5])
                
                # Display ON
                if display == 'on' and store_state >= 3 and K_statistic[index + 1] <= next_conf_interval[1]:
                    
                    selected_distance = K_support[index - 1]
                    selected_simulations = K_updated_simulations[:, index - 1]
                    observed_value = K_statistic[index - 1]
                    mu, sigma = np.mean(selected_simulations), np.std(selected_simulations)
                    conf_interval = np.percentile(selected_simulations, [2.5, 97.5])
                    
                    # Ensure the save directory exists
                    full_save_path = os.path.join(save_path, data_info)
                    os.makedirs(full_save_path, exist_ok=True)
                    
                    # Display the histogram of the simulations
                    plt.hist(selected_simulations, bins=30, density=True, alpha=0.6, color='g', label='Simulations')

                    # Generated a normal distribution curve on our data
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, mu, sigma)
                    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

                    # Display the observed value
                    plt.axvline(observed_value, color='r', linestyle='dashed', linewidth=2, label='Observed Value')

                    # Display the IC-95%
                    plt.axvline(conf_interval[0], color='b', linestyle='dashed', linewidth=2, label='IC-95%')
                    plt.axvline(conf_interval[1], color='b', linestyle='dashed', linewidth=2)

                    title = f"Distance {selected_distance}: mu = {mu:.2f}, sigma = {sigma:.2f}"
                    plt.title(title)
                    plt.legend()
                    
                    # Save the figure
                    file_path = os.path.join(full_save_path, 'IC_distance_detection_2nd_line.png')
                    plt.savefig(file_path)
                    plt.show()

                    # Print parameters of the normal distribution and IC-95%
                    print(f"Distance: {selected_distance}")
                    print(f"Mu (mean) = {mu}")
                    print(f"Sigma (standard deviation) = {sigma}")
                    print(f"Observed value = {observed_value}")
                    print(f"Confidant Interval of 95% = [{conf_interval[0]}, {conf_interval[1]}]")
                    
                    # Return the distance value where we first match this condition 
                    return selected_distance
                
                # Display OFF
                elif  display == 'off' and store_state >= 3 and K_statistic[index + 1] <= next_conf_interval[1]:

                    # Return the distance value where we first match this condition 
                    selected_distance = K_support[index - 1]
                    return selected_distance
            else:
                store_state = 0
        return None