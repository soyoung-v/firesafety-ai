from simulator import protocol


def test_encode_current_scales_by_ten():
    assert protocol.encode_current(5.2) == "052"


def test_encode_temperature_scales_by_ten():
    assert protocol.encode_temperature(27.2) == "272"


def test_encode_humidity_scales_by_ten():
    assert protocol.encode_humidity(48.4) == "484"


def test_encode_leak_ma_is_not_scaled():
    assert protocol.encode_leak_ma(24.0) == "24"


def test_encode_total_current_is_not_scaled():
    assert protocol.encode_total_current(33.0) == "33"


def test_encode_voltage_is_not_scaled():
    assert protocol.encode_voltage(224.0) == "224"


def test_encode_values_are_clamped_to_field_width():
    assert protocol.encode_leak_ma(150.0) == "99"
    assert protocol.encode_total_current(150.0) == "99"
    assert protocol.encode_fire_raw(-10.0) == "0000"


def test_encode_arc_counter_pads_to_four_digits():
    assert protocol.encode_arc_counter(7) == "0007"


def test_encode_door():
    assert protocol.encode_door(False) == "0"
    assert protocol.encode_door(True) == "1"


def test_build_aerror_no_bits_set():
    assert protocol.build_aerror(set(), set()) == "00000000"


def test_build_aerror_arc_bit_channel_1_sets_byte0_bit0():
    assert protocol.build_aerror({1}, set()) == "01000000"


def test_build_aerror_arc_bit_channel_5_sets_byte0_bit4():
    assert protocol.build_aerror({5}, set()) == "10000000"


def test_build_aerror_arc_bit_channel_6_sets_byte1_bit0():
    assert protocol.build_aerror({6}, set()) == "00010000"


def test_build_aerror_arc_bit_channel_10_sets_byte1_bit4():
    assert protocol.build_aerror({10}, set()) == "00100000"


def test_build_aerror_multiple_arc_channels():
    assert protocol.build_aerror({1, 3, 6}, set()) == "05010000"


def test_build_aerror_alarm_bit_leakage_sets_byte3_bit0():
    assert protocol.build_aerror(set(), {protocol.ALARM_BIT_LEAKAGE}) == "00000001"


def test_build_aerror_alarm_bit_overcurrent_sets_byte3_bit6():
    assert protocol.build_aerror(set(), {protocol.ALARM_BIT_OVERCURRENT}) == "00000040"


def test_build_aerror_byte2_device_error_always_zero():
    aerror = protocol.build_aerror({1}, {protocol.ALARM_BIT_FIRE})
    assert aerror[4:6] == "00"
