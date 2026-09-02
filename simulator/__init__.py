"""firesafety-be의 /m_noUpload.php 하드웨어 프로토콜을 재현하는 독립 Sensor Simulator.

training/scenario/*의 ScenarioEngine을 그대로 재사용해서 물리값 시계열을 만들고,
protocol.py에서 하드웨어 raw 인코딩으로 변환한 뒤 HTTP GET으로 Spring Boot에 전송한다.
firesafety-be 내부 Mock(MockSensorFrameGenerator 등)과 달리 실제 장비처럼 프로세스 밖에서
HTTP로만 데이터를 넣는다.
"""
