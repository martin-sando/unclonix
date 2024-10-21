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

    with open("../input/bot_photo.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    hash = key_generation_sandbox.handle_image()
    if (hash == "Error"):
        bot.send_message(message.from_user.id, "An error occured while calculating hash")
    else:
        bot.send_message(message.from_user.id, "Image hash is " + hash)

bot.polling(none_stop=True, interval=0)