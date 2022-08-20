# author: Eshan dated: 08/15/2022
from WOZ_Classes_A import *

print('Running....\n')
print('Hi! How are you doing today?')

# padding done to make the recent result list work and make it of an acceptable length before we can see any previous tags
res_recent.append(999)
res_recent.append(999)
res_recent.append(999)
res_recent.append(999)
res_recent.append(999)

while True:
    message = input('User: ')
    ints = Model_Helpers().predict_class(message)
    #print("Here ", Model_Helpers().predict_class(message)[0]['probability'])
    # print('ints:', ints)
    res = Model_Helpers.get_repsonse(ints, Model_Helpers.intents)
    res_recent.append(res)
    # print(res)
    print(res_recent[5:])
    count = 0
    max_from_prev_tags = max(res_recent, key=res_recent.count)
    for i in range(len(res_recent)):
        if res_recent[i] == max_from_prev_tags:
            # print(prev_tags[i],'==',max_from_prev_tags)
            count += 1
        else:
            count = 0
            continue
    if count >= 3:
        count = 0
        print(Talk_Something_Else())

    if Model_Helpers().predict_class(message)[0]['probability'] >= '.82':
        # ---------------------------------------------------------------------------------
        if res == 0:
            print('Hi! How are you?')
        # --------------------------------------------------------------------------------
        elif res == 1:
            print("That's great!")
        # --------------------------------------------------------------------------------
        elif res == 2:
            print('I am doing good! Thanks for asking!')
        # --------------------------------------------------------------------------------
        elif res == 3:
            print('It has been good so far. I am really happy here and pretty satisfied with my given condition! So many amenities here '
                  'that really makes me feel that I have come to a better place and the future will be bright!')
        elif res == 28 and res_recent[-2] == 3:
            print('As I have said, the place is really good. People are great here. Given the fact how good the univerisity is '
                  ', I am sure I will do great in the future!')
        # ---------------------------------------------------------------------------------
        elif res == 4 or (res == 4 and 'mexico' in message):
            PlayVideo("VideoClips/Origin03.mp4")
            print("Growing up was kind of like a very boring version of the Disney film Encanto. "
                  "Everybody knew each other. We all lived the same, and all cared about each other, "
                  "even if there was small city gossip. The real, non Disney magic happened every night "
                  "when everybody tuned their radio to the same frequency and the city danced. "
                  "The music went away when the cartel became strong, and that happened when I was about 16")
        elif res == 28 and res_recent[-2] == 4:
            PlayVideo("VideoClips/Origin04.mp4")
            print('Due the government being corrupt and the crimes increasing at a rapid rate, '
                  'the country was not safe at all and that is why my mom and dad decided '
                  'to send me out to the States so that I could pursue my dreams with no fear.'
                  'I had very little connection with the cartel. We paid our taxes for the most part, '
                  'and stayed clear. That’s what 99.9 percent of people do. I only heard rumors '
                  'and then those rumors became true, and the city became unlivable due to opportunity and '
                  'safety.')
        # ---------------------------------------------------------------------------------
        elif res == 5:
            print('My Dads name is Sergio. My dad was a teacher, just like my mom.')
        elif (res == 25 and res_recent[-2] == 5) or (res == 25 and res_recent[-3] == 5) or (
                res == 25 and res_recent[-4] == 5):
            print('He died when I was in school at the age of 42.')
        elif (res == 26 and res_recent[-2] == 5) or (res == 26 and res_recent[-3] == 5) or (
                res == 26 and res_recent[-4] == 5):
            print('His name is Sergio.')
        elif (res == 28 and res_recent[-2] == 5) or (res == 28 and res_recent[-3] == 5) or (
                res == 28 and res_recent[-4] == 5):
            print('My dad was a really smart and a hardworking person - really resourceful, '
                  'and a really caring man! He loved my mom and his family like anything!')
        # ---------------------------------------------------------------------------------
        elif res == 6:
            PlayVideo("VideoClips/Work01.mp4")
            print("At approximately twenty-two years old I sadly needed to drop out due "
                  "to an illness in my family. My father passed, and my mother could not "
                  "feed all the mouths in our home. There I took up working as a mechanic on old "
                  "trucks, which I found fulfilling and just as intellectually stimulating as "
                  "my time in chemical engineering. Working with your "
                  "hands sometimes connects you with your mind more than you think!")
        elif res == 28 and res_recent[-2] == 6:
            PlayVideo("VideoClips/Work02.mp4")
            print("I had worked in Mexico for minor jobs for approximately two years after I dropped out of college. I heard that "
                  "there were more opportunities in United States working with Celanese Corporation of America which "
                  "is a big, large company here in the United States. People told me that if I crossed the border "
                  "there would be no issues for an engineer like myself to do small jobs… and it would be safe. "
                  "Sadly there was no work when I arrived. So I did what most migrants do, working fields, "
                  "delivering goods, working as a roofer. Any small job that pays the bills for my family. "
                  "But it is like waves, they come and go, but luckily are consistent… even if they give no benefits.")
        elif res == 28 and res_recent[-3] == 6:
            PlayVideo("VideoClips/Work04.mp4")
            print("Most Chicanos claim that Experiences are the best education and although experiences and "
                  "I agree with them but I want, I'd rather see a total educated man that has experienced "
                  "working with the people and the people traits, but also what you call a formal education. "
                  "By going to school and reading and researching and learning. That to me is a total education "
                  "and knowing two languages and knowing how to read and write both languages to be the best "
                  "of one's ability. That's a total man to me. And not only that a Chicano knows his "
                  "experiences in life, this does not complete a total man and therefore this is my idea, "
                  "so although I dropped out of school, school continued. That's why I educated myself after"
                  " I was forced to leave school, because I knew I could work as hard as any man around from "
                  "sunup to sundown and just like my dad, nobody could beat him at work.")
        # ---------------------------------------------------------------------------------
        elif res == 7:
            PlayVideo("VideoClips/Work03.mp4")
            print("I work very hard. I wake up at 5am every day to do odd jobs, and "
                  "finish work at 7pm. Living on 19 thousand dollars a year gives me very little "
                  "room for free time. I just work, and have Sundays with my daughter Rose.")
        elif (res == 29 and res_recent[-2] == 7):
            print('I am pursuing my Masters from Vanderbuilt University in Nashville, Tennessee.')
        elif (res == 3 and res_recent[-2] == 7) or (res == 3 and res_recent[-3] == 7) or (
                res == 28 and res_recent[-2] == 7):
            print('I love it so far! I mean really good faculty, '
                  'made tonnes of new friends and the college has some really good extra-curricular activities.')
        # --------------------------------------------------------------------------------
        elif res == 8:
            print('I am currently talking to you in this here museum')
        elif (res == 41 and res_recent[-1] == 8) or (res == 41 and res_recent[-2] == 8) or (
                res == 41 and res_recent[-3] == 8):
            print("Oh, I am currently living in Phoenix")
        elif (res == 28 and res_recent[-1] == 8) or (res == 28 and res_recent[-2] == 8) or (
                res == 28 and res_recent[-3] == 8):
            PlayVideo("VideoClips/Geography01.mp4")
            print("Well I am here in Phoenix currently, I love the mountains, but hate the heat. "
                  "I did not think moving north would get hotter, but jobs are here. Sometimes on the "
                  "Sundays I have off my daughter and I go to south mountain where she runs up the rocks. "
                  "I don’t know if there are any careers in rock climbing, but she has a chance if there are.")
        # ---------------------------------------------------------------------------------
        elif res == 9:
            PlayVideo("VideoClips/Dreams01.mp4")
            print('I am pretty positive about it! It is my dream that I am apart of this society. '
                  'I could not be apart of my own country, Mexico, because it was not safe. So it is '
                  'my dream that I am apart of this community. That I contribute to it so I might make a '
                  'better world for my daughter.')
        elif (res == 28 and res_recent[-2] == 9):
            PlayVideo("VideoClips/Dreams02.mp4")
            print('Well I am a huge car nerd. Of course, I once worked as a mechanic. '
                  'It is my dream to have a 1998 Honda Integra Type R. Now I know that is not the '
                  'most flashy car but you can do so many modifications, and if I am lucky, one day I '
                  'can afford it')
        elif (res == 28 and res_recent[-3] == 9):
            print('I mean whats more to say!')
        # ---------------------------------------------------------------------------------
        elif res == 10:
            print('I have a daughter Rose, with my wife of 10 years, Carla')
        elif (res == 32 and res_recent[-2] == 10) or (res == 32 and res_recent[-3] == 10) or (
                res == 32 and res_recent[-4] == 10):
            print('Rose is 8 years old as of now')
        elif (res == 33 and res_recent[-2] == 10) or (res == 33 and res_recent[-3] == 10) or (
                res == 33 and res_recent[-4] == 10):
            PlayVideo("VideoClips/Family02.mp4")
            print("My gosh I love my daughter. She is the greatest thing in my life. "
                  "She is eight years old now, and  is obsessed with paw patrol, which I can not stand. "
                  "But I watch it after I get home from work to see the smile on her face. I do like "
                  "the character Rocky though")
        elif (res == 28 and res_recent[-2] == 10) or (res == 28 and res_recent[-3] == 10) or (
                res == 28 and res_recent[-4] == 10):
            PlayVideo("VideoClips/Children01.mp4")
            print("I am finding it increasingly difficult to raise Rose. I am educated but my English is "
                  "terrible, and I would rather spend the little time I have not working, having a "
                  "relationship with my daughter. I am happy she is here where the opportunity is, but "
                  "there are sacrifices. You trade your home for opportunity and safety. That’s all I want "
                  "for her, but her opportunities are hard for me to navigate.")

        # elif (res == 37 and res_recent[-2] == 10) or (res == 37 and res_recent[-3] == 10) or (res == 37 and res_recent[-4] == 10):
        #    print('As I have said before, they go to the Nashville Elementary School')
        # ---------------------------------------------------------------------------------
        elif res == 11:
            print('I love to spend time with my wife and my daughter and go out on picnics with them')
        elif (res == 28 and res_recent[-2] == 11):
            print('It is really good!')
        elif (res == 28 and res_recent[-3] == 11):
            print('Thats all I want to say - thats pretty much it!')
        elif (res == 34 and res_recent[-2] == 11) or (res == 34 and res_recent[-3] == 11) or (
                res == 34 and res_recent[-4] == 11):
            print('I have been doing it for a long time now! I never get bored of it!')
        # --------------------------------------------------------------------------------
        elif res == 12:
            print('Bye! Nice talk to you! Have a great day and stay safe!')
            break
        # --------------------------------------------------------------------------------
        elif res == 13:
            print('Yes I sure do! Initially, I did use to miss them a lot! '
                  'But one slowly gets used to it. I try my best to stay in touch with my by calling her every evening.')
        elif (res == 35 and res_recent[-2] == 13) or (res == 35 and res_recent[-3] == 13) or (
                res == 35 and res_recent[-4] == 13):
            print('Normally I FaceTime them, or WhatsApp call them')
        elif (res == 36 and res_recent[-2] == 13) or (res == 36 and res_recent[-3] == 13) or (
                res == 36 and res_recent[-4] == 13):
            print('As I have said before, once everyday')
        # -------------------------------------------------------------------------------
        elif (res == 14):
            PlayVideo("VideoClips/Culture02.mp4")
            print("My family is known for chilaquiles which are a very traditional type of breakfast. "
                  "They are fried tortillas swimming in a red or green spicy sauce and topped with sour cream, "
                  "cheese, and some fresh onion. But what we do is combine them with grilled steak, egg, or "
                  "chicken. My mother would combine bollilo to make a torta de chilaquiles, which is this "
                  "amazing sandwich that every barrio will tell you they do best. But the truth is my mother "
                  "does them best!")
        elif (res == 43 and res_recent[-2] == 14) or (res == 43 and res_recent[-3] == 14) or (
                res == 43 and res_recent[-4] == 14):
            print('It is spicy and it is lovely!')
        elif res == 44 or (res == 44 and res_recent[-2] == 14) or (res == 44 and res_recent[-3] == 14) or (
                res == 44 and res_recent[-4] == 14) or (
                res == 44 and res_recent[-5] == 14) or (res == 44 and res_recent[-6] == 14):
            print('Back in my country we have amazing dishes! '
                  'We have Licuados, agua fresca, burritos, tacos, beans and rice etc.! We have a lot!')
        elif (res == 51 and res_recent[-2] == 44) or (res == 51 and res_recent[-3] == 44) or (
                res == 51 and res_recent[-4] == 44):
            print("Yes I do. My favorite is agua fresca and my mom's torta da chilaquiles")
        elif (res == 51 and res_recent[-2] == 14) or (res == 51 and res_recent[-3] == 14) or (
                res == 51 and res_recent[-4] == 14):
            print('I just said, I do!')
        # -------------------------------------------------------------------------------
        elif res == 15:
            PlayVideo("VideoClips/Culture04.mp4")
            print("Well for us everything revolves around the church. In Mexico and here, the church is our "
                  "first home and being Catholic is very important to us. I has always been my dream to "
                  "take my daughter to Templo de Santo Domingo in Oaxaca")
        elif (res == 28 and res_recent[-1] == 15) or (res == 28 and res_recent[-2] == 15):
            PlayVideo("VideoClips/Culture01.mp4")
            print("My favorite thing about my culture is the Licuados. In Mexico, you'll "
                  "recognize these stands by the big glass jars on display, filled with all kinds of "
                  "chopped fruit. Licuados are fruit shakes made with like a milk. They come in so many "
                  "styles but banana-chocolate, strawberry, mamey (an orange fruit with a texture similar "
                  "to avocado) are the best. If you want something lighter, I always ask for an agua "
                  "fresca: the same blends of fruit, but without milk. On summer days, the greatest thing "
                  "is an agua fresca with the person you care about.")
        elif (res == 56):
            PlayVideo("VideoClips/Culture03.mp4")
            print("I love dancing. I grew up learning a very formal dance we call La Conquista. "
                  "La Conquista (the Conquest) is a traditional Mexican dance that, as the name suggests "
                  "even if you don’t know spanish, tells the story of the Spanish conquest. Masked dancers "
                  "play all the key historical players, from the conquistador Hernán Cortés and La Malinche, "
                  "a woman who acted as his interpreter and adviser, to the Aztec ruler Moctezuma. "
                  "It depicts the death of Hernán Cortés and La Malinche. It’s particularly popular but"
                  "more of a performance than a popular dance. But for popular dances I love to Cumbia! I "
                  "just love the rhythm")
        # ---------------------------------------------------------------------------------
        elif res == 16:
            print(Gratitude())
        # ---------------------------------------------------------------------------------
        elif res == 17:
            print('Thanks!')
        # ---------------------------------------------------------------------------------
        elif res == 18:
            print('Alright')
        # ---------------------------------------------------------------------------------
        elif res == 19:
            print('My family and I practice the religion of Christianity. We are Catholics.')
        # --------------------------------------------------------------------------------
        elif res == 20:
            PlayVideo("VideoClips/Origin01.mp4")
            print('My name is Juan Carlos and I was born the '
                  '20th of October, 1986, in a small village in Piedras Negras, Coahuila, Mexico. '
                  'I lived there most of my life until I was approximately twenty-five years old, then I '
                  'immigrated here.')
        elif (res == 28 and res_recent[-1] == 20) or (res == 28 and res_recent[-2] == 20):
            PlayVideo("VideoClips/Origin02.mp4")
            print('I migrated to the United States due to cartel and a family member who took a small loan '
                  'and was unable to pay it back. Cartel see the entire family as part of that debt, '
                  'so it was unsafe for myself, wife, and daughter. So we made the trip here')
        elif (res == 55) or (res == 55 and res_recent[-1] == 20) or (res == 55 and res_recent[-2] == 20) or (
                res == 55 and res_recent[-3] == 20):
            PlayVideo("VideoClips/Origin04.mp4")
            print(
                'I had very little connection with the cartel. We paid our taxes for the most part, and stayed clear. '
                'That’s what 99.9 percent of people do. I only heard rumors and then those rumors became true, '
                'and the city became unlivable due to opportunity and safety.')
        # ------------------------------------------------------------------------------
        elif res == 21:
            PlayVideo("VideoClips/Family01.mp4")
            print("Well it is me, my daughter Rose who is eight, my wife who is named Carla who is my "
                  "age, here in the USA. My father died while I attended college, and my Mother Violeta "
                  "is in Mexico with the rest of my family. Sadly, I cannot see them because of immigration. ")
        # ------------------------------------------------------------------------------
        elif (res == 22):
            print(
                'Her name is Violeta and I love her! She used to be a teacher and now she is retired. She is the best person i have known so far and she is the reason what I am what I am right now! '
                'She really took care of me and did a great job by pushing me through all the tough times and the hard times! I do miss her and my dad too!')
        # -------------------------------------------------------------------------------
        elif res == 23:
            print(
                'My father used to work as a teacher like my mom. My mom used to be a teacher and now she is retired.')
        # -------------------------------------------------------------------------------
        elif res == 24:

            print('Sadly there was no work when I arrived. So I did what most migrants do, '
                  'working fields, delivering goods, working as a roofer. Any small job that pays the '
                  'bills for my family. But it is like waves, they come and go, but luckily are consistent, '
                  'even if they give no benefits.')
        # --------------------------------------------------------------------------------
        elif res == 27:
            print(
                'There is a huge difference. It is a lot safer. There are multiple amenities and adjustments for immigrants!'
                ' Also, I feel safe here!')
        elif (res == 28 and res_recent[-2] == 27):
            print(
                'The community is good and at least I dont have to worry about crimes and problems like wars and everything')
        # --------------------------------------------------------------------------------
        elif res == 40:
            print('I am doing my masters from Vanderbuilt University')
        elif (res == 28 and res_recent[-2] == 40) or (res == 28 and res_recent[-3] == 40) or (
                res == 28 and res_recent[-4] == 40):
            print(
                'I love the university and the experience is great! The university has many extra curricular activities '
                'and has many inter departmental courses tha I love to take.')
        elif (res == 3 and res_recent[-2] == 40) and (res == 3 and res_recent[-3] == 40) or (
                res == 3 and res_recent[-4] == 40):
            print('The experience has been great so far. The faculty is great and the courses are amazing!')
        # ----------------------------------------------------------------------------------
        elif res == 30:
            print('My dads name was Sergio and my moms name is Violetta. I love them a lot and being here, '
                  'I miss them sometimes! My father died while I attended college, and my Mother Violeta '
                  'is in Mexico with the rest of my family. Sadly, I cannot see them because of immigration.')
        # ----------------------------------------------------------------------------------
        elif res == 31:
            print('I am originally from Mexico.')
        elif res == 35:
            print('I usually call them via FaceTime or Whatsapp Call')
        # --------------------------------------------------------------------------------
        elif res == 38:
            PlayVideo("VideoClips/Love02.mp4")
            PlayVideo("VideoClips/Love03.mp4")
            # GLITCH HERE!!!!!!!!!!!!
            print('Her name is Carla and we have been married for 10 years now and we have a daughter together.'
                  'We got married in 2013, the wedding was small but it was one of the best moments in my life. '
                  'We were living close, and got married at St. Augustine church. Nobody was there but us, '
                  'but it was nice that way. We could truly look at each other.'
                  'We still try to go on dates, her sister lives in the area so she takes Rose, and we like to go dancing. '
                  'She is a great dancer, but I think I am slightly better if the music pics up. '
                  'She will tell you the opposite.')

        elif (res == 39 and res_recent[-2] == 38) or (res == 39 and res_recent[-3] == 38) or (
                res == 39 and res_recent[-3] == 38):
            print('She is 36 years old, same as me')
        # ---------------------------------------------------------------------------------
        elif res == 42:
            print(
                'I like rock, pop and hip hop! My favorite song is Sometimes I feel like screaming by Deep Purple')
        elif (res == 28 and res_recent[-2] == 42):
            print('Because I listened to it while growing up so I like it!')
        # ---------------------------------------------------------------------------------
        elif res == 41:
            print('I live in Phoenix, Arizona.')
        # -----------------------------------------------------------------------------------
        elif res == 45:
            print('I would like to work for Dodge, Jeep, Chrysler, any leading '
                  'car manufacturing company, to be honest. Even Sabelt would work!')
        # ---------------------------------------------------------------------------------------
        elif res == 46:
            print('Given the fact that most car manufacturing companies are in Detroit, '
                  'I would not mind relocating, plus it is always more exposure I am after!')
        # --------------------------------------------------------------------------------------
        elif res == 40:
            print('I am doing my Masters from Vanderbuilt University')

        elif (res == 3 and res_recent[-2] == 40):
            print('I love it so far! I mean really good faculty, '
                  'made tonnes of new friends and the college has some really good extra-curricular activities.')

        elif (res == 51 and res_recent[-2] == 40) or (res == 51 and res_recent[-3] == 40):
            print('I love it so far! I mean really good faculty, '
                  'made tonnes of new friends and the college has some really good extra-curricular activities.')

        elif (res == 47):
            print('I like horror and Sci-Fi. My favorite movie is Interstellar.')

        elif (res == 28 and res_recent[-2] == 47) or (res == 28 and res_recent[-3] == 47):
            print('Since I always watched these kind of movies when I was growing up.')

        elif (res == 48):
            PlayVideo("VideoClips/Origin02.mp4")
            print('I migrated to the United States due to cartel and a family member who took a small'
                  ' loan and was unable to pay it back. Cartel see the entire family as part of that debt,'
                  ' so it was unsafe for myself, wife, and daughter. So we made the trip here')
        elif res == 49:
            print('Alfred, Bart, Casey, they go to the Nashville Elementary School. Martina is toddler so she stays at home.')
        # ------------------------------------------------------------------------------------
        elif res == 50:
            print('A community that takes care of all its members, is a good community in my book! '
                  'A community that is secular and treats all the people fairly is a good community!')

        elif res == 52:
            PlayVideo("VideoClips/Family03.mp4")
            print('My mother is the greatest influence on my life. When I was my daughters age, '
                  'she would bring home math problems she procured from the high school teacher, '
                  'and force me to sit in the silence until I could come up with an answer. When I finished, '
                  'regardless if I got it right or not, she gave me her famous tamales. If I got '
                  'the answer right, I was rewarded with the sauce Christmas style. Which is when you '
                  'mix red and green. It’s the best.')

        elif (res == 28 and res_recent[-2] == 52) or (res == 28 and res_recent[-3] == 52) or (
                res == 28 and res_recent[-4] == 52):
            print('I have given all my reasons to be honest!')

        elif res == 53:
            print('Great! We have the same taste in that way!')

        elif res == 54:
            print('Go ahead ask me a question. Just type exit whenever you are done')
            ques = ''
            while True:
                ques = input('Type: ')
                print(nlp(question=ques, context=context)['answer'])
                choice = input(
                    'Do you want to ask more in depth questions? Type "y" or "yes" to continue else type "exit".')

                if choice == 'y' or choice == 'yes':
                    continue
                else:
                    print('Alright moving normal mode now!')
                    break
        elif res == 57:
            PlayVideo("VideoClips/Children01.mp4")
            print('I am finding it increasingly difficult to raise Rose. I am educated but my English is terrible, '
                  'and I would rather spend the little time I have not working, having a relationship with my '
                  'daughter. I am happy she is here where the opportunity is, '
                  'but there are sacrifices. You trade your home for opportunity and safety. '
                  'That’s all I want for her, but her opportunities are hard for me to navigate.')
        elif res == 58:
            PlayVideo("VideoClips/Work04.mp4")
            print("Most Chicanos claim that Experiences are the best education and although experiences and "
                  "I agree with them but I want, I'd rather see a total educated man that has experienced "
                  "working with the people and the people traits, but also what you call a formal education. "
                  "By going to school and reading and researching and learning. That to me is a total education "
                  "and knowing two languages and knowing how to read and write both languages to be the best "
                  "of one's ability. That's a total man to me. And not only that a Chicano knows his "
                  "experiences in life, this does not complete a total man and therefore this is my idea, "
                  "so although I dropped out of school, school continued. That's why I educated myself after"
                  " I was forced to leave school, because I knew I could work as hard as any man around from "
                  "sunup to sundown and just like my dad, nobody could beat him at work.")
        elif res == 59:
            PlayVideo("VideoClips/Geography02.mp4")
            print("I commute a lot here. Because I do odd jobs we own a truck, "
                  "but the gas prices are killing us so we don’t have much time to travel. "
                  "We have gone to Sedona, and Flagstaff for picnics. But no extra time to see the "
                  "geography with gas at this price")
        elif res == 60:
            PlayVideo("VideoClips/Love01.mp4")
            print("It was late June of 2012. I was enjoying the summer working here. The day started out "
                  "normally enough. I went to my job with the roofing contractor, where I laid adhesive. "
                  "I worked from 5 am until about noon, and took a break because it was to hot for us to work. "
                  "I then went home, ate lunch, and then went to get something to eat because lunch was to "
                  "small. I went to the market down the street and the woman in the deli was not beautiful "
                  "in the magazine sense, but full of light drew me in. I was so drawn to her, that instead "
                  "of asking for sliced meats, I put my hand out to shake and introduce myself. She looked "
                  "at me like I was crazy, smiled, and shook my hand. I left and we exchanged glances. "
                  "Then, of course I came back to the deli every day for a month to talk to her, and "
                  "order food I could not afford until I worked up the never to ask her on a date. "
                  "She said, yes and the rest is history.")
        elif res == 61:
            PlayVideo("VideoClips/Family04.mp4")
            print("I did not know my grandparents well. They were quiet, hardworking, and went to church. "
                  "My grandmother was always making food, and my grandfather was always making trouble in "
                  "the kitchen.")
        elif res == 62:
            PlayVideo("VideoClips/School01.mp4")
            print("The time that I lived in Mexico, all my formal schooling took place there: my grade "
                  "school, high school, and I even went to college. This is very rare for my community "
                  "due to the poverty, but I excelled in math. I started a degree in chemical engineering "
                  "in the University of Mexico. At approximately twenty-two years old I sadly needed to "
                  "drop out due to an illness in my family. My father passed, and my mother could not feed "
                  "all the mouths in our home. There I took up working as a mechanic on old trucks, which I "
                  "found fulfilling and just as intellectually stimulating as my time in chemical engineering. "
                  "Working with your hands sometimes connects you with your mind more than you think!")

    else:
        print('Sorry I dont understand, please rephrase it for me, or better, ask me something else!')
        ques = input('Type: ')
        if ques == 'no' or ques == 'leave it' or ques == 'moving on':
            print('Alright')
        else:
            print(nlp(question=ques, context=context)['answer'])
print('Appeared Tags: ', res_recent[5:])

