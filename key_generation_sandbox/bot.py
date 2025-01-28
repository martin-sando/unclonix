# Bot used to handle photos. To get bot key and bot telegram link, contact @Alex2184
# Alternatively, create and use your own key and link
# Right now, bot only supports photos. To use it, send photo and it will return hash.
# It also may be executed in key_generation_sandbox
import datetime
from pathlib import Path
import telebot
import traceback
import key_generation_sandbox
import utils


with open(Path.home() / ".ssh" / "unclonix" / "unclonix_hash_bot.txt") as f:
    key = f.readline().strip()
bot = telebot.TeleBot(key)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.from_user.id, "To get started, send label image, or type /help for more info")

@bot.message_handler(commands=['help'])
def manual(message):
    bot.send_message(message.from_user.id, "This is a bot to convert Unclonix label images to hash.\nRight now it "
                                           "supports only photos and converts them to hash.\nUnclonix website: "
                                           "https://unclonix.com/")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    print('message.photo =', message.photo)
    file_id = message.photo[-1].file_id
    print('file_id =', file_id)
    file_info = bot.get_file(file_id)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    name = "bot" + datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S%f') + ".jpg"
    with open("../input/" + name, 'wb') as new_file:
        new_file.write(downloaded_file)
    utils.set_total_time()
    try:
        the_hash = key_generation_sandbox.run(name)
        bot.send_message(message.from_user.id, "Image hash is " + the_hash)
    except:
        print(traceback.format_exc())

@bot.message_handler(content_types=['document'])
def handle_photo(message):
    print('message.photo =', message.photo)
    file_id = message.document.file_id
    print('file_id =', file_id)
    file_info = bot.get_file(file_id)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    name = "bot" + datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S%f') + ".jpg"
    with open("../input/" + name, 'wb') as new_file:
        new_file.write(downloaded_file)
    utils.set_total_time()
    try:
        the_hash = key_generation_sandbox.run(name)
        bot.send_message(message.from_user.id, "Image hash is " + the_hash)
    except:
        print(traceback.format_exc())


def run():
    print('Running bot')
    bot.polling(none_stop=True, interval=0)


if __name__ == '__main__':
    run()
