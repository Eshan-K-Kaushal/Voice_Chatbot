# ALL are random weights
# I only did it for 3 nodes - just for testing, but they have 10 kids in total
# The CODE GIVES SUGGESTIONS on the basis of the max weights associated to a child node from the current node
import numpy as np

# working with only 10 tags

dict_tags = { 'A':{1:'experience in new count'}, 'B': {2: 'experience in old country'}, 'C':{3:'dads job'}, 'D':{4:'how I support myself'}, 'E':{5:'like_work'},
             'F':{6:'location'}, 'G':{7:'What I think of my future'},
              'H':{8:'about my kids'}, 'I':{9:'fave_activity'}, 'J':{10:"Let's say goodbye and call it a day for now."}}

ls_succ = []

# graphs
graph = { "A": {"A": 1, "B": 8, "C": 5, "D": 7, "E": 4, "F": 0, "G": 6, "H": 3, "I": 2, "J": -1},
          "B": {"A": 10, "B" : 1, "C" : 7, "D" : 4, "E" : 8, "F" : 2,"G" : 9,"H" : 3,"I" : 6,"J" : 0},
        "C": {"A": 6, "B" : 2, "C" : 0, "D" : 9, "E" : 3, "F" : 8,"G" : 7,"H" : 10,"I" : 1,"J" : 5}
        }

def give_sugg(node):

    #node = "'"+node+"'"

    print('We just talked about: ', list(dict_tags[str(node)].values())[0])
    for i in graph:
        if i == str(node):
            a = (list(graph[i].values()))
    #print(a)
    arr_a = np.array(a)
    #print(arr_a)

    asc_arr_a = np.argsort(arr_a)
    arr_a_new = arr_a[asc_arr_a]
    #print(arr_a_new)

    sugg_list = []
    sugg_list.append(arr_a_new[len(arr_a_new)-1])
    sugg_list.append(arr_a_new[len(arr_a_new)-2])

    print('Max vals are - the suggestions are: ', sugg_list)

    reverse_dict = dict()

    for key in graph.keys():
        reverse_dict[key] = dict()
        nested_dict = graph[key]
        for k in nested_dict.keys():
            val = nested_dict[k]
            if val in reverse_dict[key]:
                reverse_dict[key][val].append(k)
            else:
                reverse_dict[key][val] = [k]

    suggs = []

    for i in sugg_list:
        #print(i)
        suggs.append((reverse_dict[str(node)][i])[0])
    print(suggs)
    seggs = []
    for i in suggs:
        seggs.append(list(dict_tags[i[0]].values())[0])
    print('Lets talk about: ',seggs)



while True:
    print('!!!!!Please read carefully!!!!!!')
    print('---------------------------------------------------------------------------------------------------------')
    print('Welcome to the suggestion system - beta. Please type the inputs for the nodes, when prompted, in CAPITAL.')
    print('----------------------------------------------------------------------------------------------------------')
    user_input_1 = input('What node are you on? ')
    user_input = "'"+user_input_1+"'"
    print(user_input)
    #give_sugg(user_input)
    give_sugg(str(user_input_1))

    choice = input('Do you wish to continue: type Y / N ')
    if choice == 'N' or choice == 'n':
        break