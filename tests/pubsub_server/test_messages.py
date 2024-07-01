from pubsub_server.messages import ImageMessage, TickMessage, Image, Tick, Message


def _random_image() -> bytes:
    return b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"


def test_image_message() -> None:
    image = _random_image()
    message = ImageMessage(data=Image(image=image))
    message_json = message.model_dump_json()
    assert message_json == f'{{"data":{{"image":"{image.hex()}"}}}}'
    message2 = ImageMessage.model_validate_json(message_json)
    assert message == message2


def test_tick_message() -> None:
    tick = 123
    message = TickMessage(data=Tick(tick=tick))
    message_json = message.model_dump_json()
    assert message_json == f'{{"data":{{"tick":{tick}}}}}'
    message2 = TickMessage.model_validate_json(message_json)
    assert message == message2


def test_tell_different_messages() -> None:
    image = _random_image()
    image_message = ImageMessage(data=Image(image=image))
    image_message_json = image_message.model_dump_json()

    possible_image_or_tick_message = Message[Tick | Image].model_validate_json(
        image_message_json
    )
    assert isinstance(possible_image_or_tick_message.data, Image)

    tick = 123
    tick_message = TickMessage(data=Tick(tick=tick))
    tick_message_json = tick_message.model_dump_json()

    possible_image_or_tick_message = Message[Tick | Image].model_validate_json(
        tick_message_json
    )
    assert isinstance(possible_image_or_tick_message.data, Tick)
