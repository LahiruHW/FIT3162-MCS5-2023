# EVALUATE (TESTER 1)

# Import the necessary global modules
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import imageio.v2 as imageio
from os import remove 

##########################################################################################################
##########################################################################################################
####################################  IMPORT YOUR FUNCTIONS HERE  ########################################

# Implentations of two block-ciphers, SPECK and SIMON, are provided for you below.

import speck as speck
from simon import SimonCipher

##########################################################################################################
##########################################################################################################
###################################  SETTING UP THE NEURAL MODELS  #######################################

csv_file = open('results.csv','w');
csv_file.write('');
csv_file.write('Accuracy,TPR,TNR,MSE,High Random\n');
csv_file.close();

# load distinguishers
json_file = open('single_block_resnet.json','r');
json_model = json_file.read();

net5 = model_from_json(json_model);
net6 = model_from_json(json_model);
net7 = model_from_json(json_model);
net8 = model_from_json(json_model);

net5.load_weights('net5_small.h5');
net6.load_weights('net6_small.h5');
net7.load_weights('net7_small.h5');
net8.load_weights('net8_small.h5');

##########################################################################################################
##########################################################################################################
##########################################################################################################

def evaluate(net,X,Y): # evaluating section
    """
    Evaluates the neural distinguisher on the data (X,Y).
    X: input data
    Y: labels
    """
    csv_file = open('results.csv','a');
    Z = net.predict(X,batch_size=10000).flatten();
    # print("z:", Z)
    Zbin = (Z > 0.5);
    # print("zbin:", Zbin)
    # j1 = np.array(Zbin);
    # np.savetxt("Zbin_value.csv", j1, delimiter=",");
    diff = Y - Z                                # difference between the labels and the predictions
    mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;                # accuracy
    tpr = np.sum(Zbin[Y==1]) / n1;              # true positive rate
    tnr = np.sum(Zbin[Y==0] == 0) / n0;         # true negative rate
    mreal = np.median(Z[Y==1]);                 # median of real pairs
    high_random = np.sum(Z[Y==0] > mreal) / n0; # percentage of random pairs with score higher than median of real pairs
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    csv_file.write(str(acc)+','+str(tpr)+','+str(tnr)+','+str(mse)+','+str(high_random)+'\n');
    csv_file.close();
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);

def plot_multiple_bar_graph(from_filename, title = 'Results', save_as="results.png"):
    """
    Plot the graphs of the results stored in file_name.
    PRECONDTION: The file must be a CSV file.
    """

    fig , ax = plt.subplots()
    ax.grid( which = "both" , zorder=0)
    ax.minorticks_on()

    # read the data from the file and close it
    csv_file = open(from_filename,'r');
    lines = csv_file.readlines();
    csv_file.close();

    accuracy, tpr , tnr , mse , high_random = [] , [] , [] , [] , [];

    for line in lines[1:]:
        data = line.split(',');
        accuracy += [ float(data[0]) ];
        tpr += [ float(data[1]) ];
        tnr += [ float(data[2]) ];
        mse += [ float(data[3]) ];
        high_random += [ float(data[4]) ];

    N = len(accuracy);
    ind = np.arange(N);
    width = 0.15;
    empty = [1.0] * N; # used to create empty space between bars

    bar1 = plt.bar(ind              , accuracy      , width , color = 'r');
    bar2 = plt.bar(ind + width      , tpr           , width , color = 'g');
    bar3 = plt.bar(ind + 2*width    , tnr           , width , color = 'b');
    bar4 = plt.bar(ind + 3*width    , mse           , width , color = 'y');
    bar5 = plt.bar(ind + 4*width    , high_random   , width , color = 'c');
    bar6 = plt.bar(ind + 5*width    , empty         , width , color = 'w');

    plt.title(title);
    plt.grid(True);

    plt.legend(
        (bar1,bar2,bar3,bar4,bar5) ,
        ('Accuracy (%)','True Positive Rate (TPR)','True Negative Rate (TNR)','Mean Squared Error (MSE)','High Random') ,
        prop = {'size': 7} ,
        # bbox_to_anchor=(0.95, 1.05)
    );

    # plt.xticks( (np.arange(4) + 2*width) , ('5','6','7','8') );

    plt.xlabel('Number of Rounds (5 , 6 , 7 , 8)');
    plt.ylabel('Normalized Value');
    
    ax.xaxis.set_ticklabels([])


    plt.savefig(save_as);
    
    # clear our al the data in the CSV file except the first line
    file = open("results.csv", 'w')
    file.write("")
    file.write("Accuracy,TPR,TNR,MSE,High Random\n")
    file.close()

##########################################################################################################
##########################################################################################################
##########################################################################################################

NUM_SAMPLES = 10**6

X5,Y5 = speck.make_speck_train_data( n = NUM_SAMPLES , nr = 5 );
X6,Y6 = speck.make_speck_train_data( n = NUM_SAMPLES , nr = 6 );
X7,Y7 = speck.make_speck_train_data( n = NUM_SAMPLES , nr = 7 );
X8,Y8 = speck.make_speck_train_data( n = NUM_SAMPLES , nr = 8 );

print("CHECKING")
X5r, Y5r = speck.real_differences_data( n = NUM_SAMPLES , nr = 5);
# print("X5R" , sp.real_differences_data(NUM_SAMPLES,5))
X6r, Y6r = speck.real_differences_data( n = NUM_SAMPLES , nr = 6);
X7r, Y7r = speck.real_differences_data( n = NUM_SAMPLES , nr = 7);
X8r, Y8r = speck.real_differences_data( n = NUM_SAMPLES , nr = 8);

print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
print('5 rounds:');
evaluate(net5, X5, Y5);
print('6 rounds:');
evaluate(net6, X6, Y6);
print('7 rounds:');
evaluate(net7, X7, Y7);
print('8 rounds:');
evaluate(net8, X8, Y8);

plot_multiple_bar_graph(
    from_filename='results.csv', 
    title='Results(speck)-normal vs. random', 
    save_as='results_speck_5_6_7_8.png'
);

print('\nTesting real differences setting now.');
print('5 rounds:');
evaluate(net5, X5r, Y5r);
print('6 rounds:');
evaluate(net6, X6r, Y6r)
print('7 rounds:');
evaluate(net7, X7r, Y7r)
print('8 rounds:');
evaluate(net8, X8r, Y8r);

plot_multiple_bar_graph(
    from_filename='results.csv', 
    title='Results(speck)-real difference', 
    save_as='results_speck_5r_6r_7r_8r.png'
);

img_1 = np.array( imageio.imread('results_speck_5_6_7_8.png') )
img_2 = np.array( imageio.imread('results_speck_5r_6r_7r_8r.png') )
imageio.imwrite('speck_comp.png' , np.concatenate((img_1 , img_2) , axis = 1) )
remove('results_speck_5_6_7_8.png')
remove('results_speck_5r_6r_7r_8r.png')

##########################################################################################################
##########################################################################################################
##########################################################################################################

NUM_SAMPLES = 10**6;  
simon5 = SimonCipher( key = 0x1918111009080100 , block_size = 32 , key_size = 64 );
simon6 = SimonCipher( key = 0x1918111009080100 , block_size = 32 , key_size = 96 );
simon7 = SimonCipher( key = 0x1918111009080100 , block_size = 32 , key_size = 128 );
simon8 = SimonCipher( key = 0x1918111009080100 , block_size = 32 , key_size = 192 );

X5,Y5 = simon5.make_simon_train_data( n = NUM_SAMPLES );
X6,Y6 = simon6.make_simon_train_data( n = NUM_SAMPLES );
X7,Y7 = simon7.make_simon_train_data( n = NUM_SAMPLES );
X8,Y8 = simon8.make_simon_train_data( n = NUM_SAMPLES );

print("CHECKING")
X5r, Y5r = simon5.real_differences_data( n = NUM_SAMPLES );
# print("X5R" , simon5.real_differences_data( NUM_SAMPLES ))
X6r, Y6r = simon6.real_differences_data( n = NUM_SAMPLES );
X7r, Y7r = simon7.real_differences_data( n = NUM_SAMPLES );
X8r, Y8r = simon8.real_differences_data( n = NUM_SAMPLES );

print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
print('5 rounds:');
evaluate(net5, X5, Y5);
print('6 rounds:');
evaluate(net6, X6, Y6);
print('7 rounds:');
evaluate(net7, X7, Y7);
print('8 rounds:');
evaluate(net8, X8, Y8);

plot_multiple_bar_graph(
    from_filename='results.csv', 
    title='Results(simon)-normal vs. random', 
    save_as='results_simon_5_6_7_8.png'
);

print('\nTesting real differences setting now.');
print('5 rounds:');
evaluate(net5, X5r, Y5r);
print('6 rounds:');
evaluate(net6, X6r, Y6r);
print('7 rounds:');
evaluate(net7, X7r, Y7r);
print('8 rounds:');
evaluate(net8, X8r, Y8r);

plot_multiple_bar_graph(
    from_filename='results.csv', 
    title='Results(simon)-real difference', 
    save_as='results_simon_5r_6r_7r_8r.png'
);

img_1 = np.array( imageio.imread('results_simon_5_6_7_8.png') )
img_2 = np.array( imageio.imread('results_simon_5r_6r_7r_8r.png') )
imageio.imwrite('simon_comp.png' , np.concatenate((img_1 , img_2) , axis = 1) )
remove('results_simon_5_6_7_8.png')
remove('results_simon_5r_6r_7r_8r.png')

##########################################################################################################
##########################################################################################################
##########################################################################################################


# concatenating all the results into one image  

img_1 = np.array( imageio.imread('speck_comp.png') )
img_2 = np.array( imageio.imread('simon_comp.png') )

imageio.imwrite('results.png' , np.concatenate((img_1 , img_2) , axis = 0) )



##########################################################################################################
##########################################################################################################
##########################################################################################################

