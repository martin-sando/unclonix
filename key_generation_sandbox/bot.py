import datetime
from pathlib import Path
import telebot
import key_generation_sandbox


with open(Path.home() / ".ssh" / "unclonix" / "unclonix_hash_bot.txt") as f:
    key = f.readline().strip()
bot = telebot.TeleBot(key)


@bot.message_handler(commands=['start', 'help'])
def manual(message):
    bot.send_message(message.from_user.id, "To get started, send label image")


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
    given_hash = key_generation_sandbox.handle_image(name)
    if given_hash == "Error":
        bot.send_message(message.from_user.id, "An error occurred while calculating hash")
    else:
        bot.send_message(message.from_user.id, "Image hash is " + given_hash)


def run():
    print('Running bot')
    bot.polling(none_stop=True, interval=0)


if __name__ == '__main__':
    run()
