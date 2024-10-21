import datetime

import telebot
import key_generation_sandbox
bot = telebot.TeleBot('7535912693:AAEpT9-dab_JXS28tAw-GTKnzYVocjc9cIs')

@bot.message_handler(commands=['start', 'help'])
def manual(message):
    bot.send_message(message.from_user.id, "To get started, send label image")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    name = "bot_photo" + str(hash(datetime.datetime.now())) + ".jpg"
    with open("../input/" + name, 'wb') as new_file:
        new_file.write(downloaded_file)
    given_hash = key_generation_sandbox.handle_image(name)
    if (given_hash == "Error"):
        bot.send_message(message.from_user.id, "An error occured while calculating hash")
    else:
        bot.send_message(message.from_user.id, "Image hash is " + given_hash)

bot.polling(none_stop=True, interval=0)