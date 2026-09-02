"""물리값 -> 하드웨어 프로토콜 raw 인코딩.

전부 firesafety-be의 SensorIngestService(파싱쪽)/MockSensorFrameGenerator(자리수쪽)에서
실제로 확인된 규칙만 사용한다 - 문서화되지 않은 자리수/스케일은 추정하지 않는다.

- volt(3자리)/s_circuit(2자리)/fire(4자리)/gas(4자리)/total_circuit(2자리)/e_energy(5자리)는
  소수점 스케일 없이 정수 그대로 인코딩한다(SensorIngestService.parseDecimal).
- tem(3자리, 부호 가능)/humi(3자리, 부호 가능)/am1~10(3자리)은 값 * 10을 인코딩한다
  (SensorIngestService.parseDecimalByTen).
- count1~10(4자리)은 회로별 누적 아크 카운터를 정수 그대로 인코딩한다.
- aerror(8자리 hex)는 byte0/1=회로별 ARC bit, byte2=DEVICE ERROR(의미 불명확이라 항상 0),
  byte3=ALARM bit(DeviceAlertService와 동일 순서: 누전/과열/습도/가스/불꽃/문열림/과전류).
"""

from __future__ import annotations

MAX_2_DIGIT = 99
MAX_3_DIGIT = 999
MAX_4_DIGIT = 9999

ALARM_BIT_LEAKAGE = 0
ALARM_BIT_OVERHEAT = 1
ALARM_BIT_HUMIDITY = 2
ALARM_BIT_GAS = 3
ALARM_BIT_FIRE = 4
ALARM_BIT_DOOR_OPEN = 5
ALARM_BIT_OVERCURRENT = 6


# 고정폭 0-padding (음수는 부호 유지 + 나머지 자리수만 padding)
def pad(value: int, width: int) -> str:
    if value < 0:
        return "-" + str(abs(value)).zfill(width)
    return str(value).zfill(width)


def encode_voltage(volt: float) -> str:
    return pad(min(round(volt), MAX_3_DIGIT), 3)


def encode_leak_ma(leak_ma: float) -> str:
    return pad(min(max(round(leak_ma), 0), MAX_2_DIGIT), 2)


def encode_temperature(temp_c: float) -> str:
    return pad(round(temp_c * 10), 3)


def encode_humidity(humidity_pct: float) -> str:
    return pad(round(humidity_pct * 10), 3)


def encode_fire_raw(fire_raw: float) -> str:
    return pad(min(max(round(fire_raw), 0), MAX_4_DIGIT), 4)


def encode_gas_raw(gas_raw: float) -> str:
    return pad(min(max(round(gas_raw), 0), MAX_4_DIGIT), 4)


def encode_total_current(total_current: float) -> str:
    return pad(min(max(round(total_current), 0), MAX_2_DIGIT), 2)


def encode_total_power(total_power: float) -> str:
    return pad(min(max(round(total_power), 0), 99999), 5)


def encode_current(current: float) -> str:
    return pad(min(max(round(current * 10), 0), MAX_3_DIGIT), 3)


def encode_arc_counter(arc_count: float) -> str:
    return pad(min(max(round(arc_count), 0), MAX_4_DIGIT), 4)


def encode_door(door_open: bool) -> str:
    return "1" if door_open else "0"


# 회로 번호(1~10) -> byte0/byte1 ARC bit 위치. DeviceAlertService.isArcBitOn과 동일 규칙.
def _arc_bit_position(channel_no: int) -> tuple[int, int]:
    byte_index = 0 if channel_no <= 5 else 1
    bit_index = (channel_no - 1) if channel_no <= 5 else (channel_no - 6)
    return byte_index, bit_index


# aerror 8자리 hex 생성. arc_channels: 이번 프레임에서 ARC bit를 켤 회로 번호 집합.
# alarm_bits: 이번 프레임에서 켤 ALARM_BIT_* 집합. byte2(DEVICE ERROR)는 의미가 불명확해 항상 0.
def build_aerror(arc_channels: set[int], alarm_bits: set[int]) -> str:
    byte0 = 0
    byte1 = 0
    byte3 = 0
    for channel_no in arc_channels:
        byte_index, bit_index = _arc_bit_position(channel_no)
        if byte_index == 0:
            byte0 |= 1 << bit_index
        else:
            byte1 |= 1 << bit_index
    for bit_index in alarm_bits:
        byte3 |= 1 << bit_index
    return f"{byte0:02X}{byte1:02X}{0:02X}{byte3:02X}"
