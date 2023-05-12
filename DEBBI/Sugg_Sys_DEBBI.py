import numpy as np

dict_tags = {
              0:'Ask me about myself - a little background',
              1:'My reason for immigration',
              2:'We can talk about my family',
              3:'Where do I live as of now',
              7:'My thoughts on US',
              8:'My experience before I moved here',
              9:'My Work',
              10:'About my father',
              11:'About my mother',
              12:"What kind of work my parents do",
              13:'What do I think of the future',
              14:'My Kids',
              15:'My favorite sport/activity',
              16:'Or we can just call it a day :)',
              17:'What I feel about my parents and how I miss them',
              18:'My favorite meal',
              19:'About my culture',
              23:'My Religion',
              24:'Change in my experience',
              26: 'My Roots',
              27:'My Kids ages',
              28:'For how long have I been doing that',
              29:'How do I stay in touch with my mom and dad',
              30:'My spouse or my relationship',
              32:'Food back in my home country',
              33:'What are my thoughts on a good community',
              36:'Importance of Education',
              37:'Trickiest part of the move'
            }

ls_succ = []

graph_2 = {0: [[1, 11], [2, 8], [4, 7], [8, 9], [7, 10], [13, 6]],
           1: [[7, 10], [8, 9], [13, 8], [4, 8], [26, 8]],
           2: [[10, 9], [11, 10], [12, 7], [14, 8], [32, 6]],
           3:[[7, 10], [26, 9], [1, 8]],
           4:[[0, 10], [1, 9], [8, 8]],
           7: [[26, 10], [28, 8], [13, 9]],
           8: [[7, 10], [9, 9], [13, 8]],
           9: [[13, 10], [33, 9], [37, 8]],
           10: [[11, 10], [12, 9], [2, 8]],
           11: [[10, 10], [12, 9], [17, 8]],
           12: [[10, 10], [11, 9], [2, 8]]
           }
states = [el for el in graph_2.keys()]

def system_suggs_bro(node):
    states = [el for el in graph_2.keys()]

    if node in states:
        #print(dict_tags[node])
        node = int(node)
        a = graph_2[node]
        scores = [el[1] for el in a]
        #print(scores)
        nodes = [el[0] for el in a]
        #print(nodes)

        max_list = sorted(zip(scores, nodes), reverse=True)[:3]
        suggs = [el[1] for el in max_list]
        fsuggs = []
        #print(suggs)
        print('We can talk about: ')
        for i in suggs:
            fsuggs.append(dict_tags[i])
        return fsuggs

    else:
        return ':)'
