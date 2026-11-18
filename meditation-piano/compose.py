"""명상 피아노 10곡 작곡·렌더링 스크립트.

각 곡은 조성·화성 진행·템포·텍스처를 정의한 SPEC으로부터 시드 고정 방식으로
생성되며, MIDI(편집용)와 MP3(감상용)로 출력된다.

    pip install numpy scipy mido lameenc
    python3 compose.py            # 10곡 전체
    python3 compose.py 3 7        # 3번, 7번만
"""
import os
import sys
import random

import numpy as np
import mido
import lameenc
from scipy.signal import fftconvolve, butter, sosfilt

SR = 44100
TARGET_SEC = 180   # 곡당 목표 길이(초)
OUT = os.path.dirname(os.path.abspath(__file__))

NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
        'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}

SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'penta': [0, 2, 4, 7, 9],
    'minpenta': [0, 3, 5, 7, 10],
}

QUALITY = {
    'maj7': [0, 4, 7, 11], 'maj9': [0, 4, 7, 11, 14], 'add9': [0, 4, 7, 14],
    'm7': [0, 3, 7, 10], 'm9': [0, 3, 7, 10, 14], 'm11': [0, 3, 7, 10, 14, 17],
    'maj7#11': [0, 4, 7, 11, 18], 'sus2': [0, 2, 7, 14], 'sus4': [0, 5, 7, 10],
    '6sus': [0, 5, 7, 9], '69': [0, 4, 7, 9, 14], 'maj': [0, 4, 7], 'm': [0, 3, 7],
    'm6': [0, 3, 7, 9], '5': [0, 7, 12],
}

# ---------------------------------------------------------------- 곡 사양
# prog: (근음, 코드 성질)   sections: (텍스처, 진행 반복 횟수, 멜로디 여부, 셈여림)
TRACKS = [
    dict(no=1, title='새벽의 호흡', en='Breath of Dawn', key='C', scale='major', bpm=60, meter=4,
         bars_per_chord=2,
         prog=[('C', 'maj7'), ('A', 'm9'), ('F', 'maj7'), ('G', '6sus')],
         sections=[('block', 1, False, 0.55), ('arp_up', 2, True, 0.7), ('arp_wave', 1, True, 0.8),
                   ('arp_up', 1, True, 0.65), ('block', 1, False, 0.5)]),
    dict(no=2, title='고요한 호수', en='Still Lake', key='D', scale='lydian', bpm=54, meter=4,
         bars_per_chord=2,
         prog=[('D', 'maj9'), ('B', 'm11'), ('G', 'maj7#11'), ('A', 'add9')],
         sections=[('pedal', 1, False, 0.5), ('arp_wave', 2, True, 0.7), ('sparse', 1, True, 0.6),
                   ('arp_wave', 1, True, 0.7), ('block', 1, False, 0.45)]),
    dict(no=3, title='달빛 정원', en='Moonlit Garden', key='A', scale='minor', bpm=58, meter=4,
         bars_per_chord=2,
         prog=[('A', 'm9'), ('F', 'maj7'), ('C', 'maj7'), ('E', 'm7')],
         sections=[('sparse', 1, False, 0.5), ('arp_up', 2, True, 0.7), ('arp_wave', 1, True, 0.8),
                   ('sparse', 1, True, 0.55), ('block', 1, False, 0.45)]),
    dict(no=4, title='숲의 숨결', en='Forest Whisper', key='G', scale='penta', bpm=64, meter=4,
         bars_per_chord=2,
         prog=[('G', 'add9'), ('E', 'm7'), ('C', 'add9'), ('D', 'sus2')],
         sections=[('pedal', 1, True, 0.5), ('arp_up', 2, True, 0.7), ('pedal', 1, True, 0.6),
                   ('arp_up', 1, True, 0.65), ('pedal', 1, False, 0.45)]),
    dict(no=5, title='흐르는 구름', en='Drifting Clouds', key='Eb', scale='major', bpm=56, meter=4,
         bars_per_chord=2,
         prog=[('Eb', 'maj7'), ('C', 'm9'), ('Ab', 'maj7'), ('Bb', 'sus4')],
         sections=[('block', 1, False, 0.5), ('arp_wave', 2, True, 0.7), ('arp_up', 1, True, 0.75),
                   ('sparse', 1, True, 0.55), ('block', 1, False, 0.45)]),
    dict(no=6, title='내면의 빛', en='Inner Light', key='E', scale='major', bpm=62, meter=4,
         bars_per_chord=2,
         prog=[('E', 'maj9'), ('C#', 'm7'), ('A', 'maj9'), ('B', 'sus4')],
         sections=[('sparse', 1, False, 0.5), ('arp_up', 2, True, 0.7), ('arp_wave', 1, True, 0.8),
                   ('arp_up', 1, True, 0.65), ('block', 1, False, 0.45)]),
    dict(no=7, title='빗소리 명상', en='Rain Meditation', key='F#', scale='dorian', bpm=66, meter=4,
         bars_per_chord=2,
         prog=[('F#', 'm9'), ('B', 'add9'), ('F#', 'm11'), ('E', '69')],
         sections=[('rain', 1, False, 0.5), ('rain', 2, True, 0.65), ('arp_wave', 1, True, 0.7),
                   ('rain', 1, True, 0.6), ('block', 1, False, 0.45)]),
    dict(no=8, title='별이 머무는 밤', en='Where Stars Rest', key='Db', scale='major', bpm=50, meter=4,
         bars_per_chord=2,
         prog=[('Db', 'maj9'), ('Bb', 'm9'), ('Gb', 'maj7#11'), ('Ab', '6sus')],
         sections=[('sparse', 1, False, 0.45), ('sparse', 2, True, 0.6), ('arp_wave', 1, True, 0.65),
                   ('sparse', 1, True, 0.5), ('block', 1, False, 0.4)]),
    dict(no=9, title='치유의 파동', en='Healing Waves', key='F', scale='major', bpm=66, meter=3,
         bars_per_chord=3,
         prog=[('F', 'maj7'), ('D', 'm9'), ('Bb', 'maj7'), ('C', 'sus4')],
         sections=[('waltz', 1, False, 0.5), ('waltz', 2, True, 0.7), ('arp_wave', 1, True, 0.75),
                   ('waltz', 1, True, 0.6), ('block', 1, False, 0.45)]),
    dict(no=10, title='평온한 귀환', en='Peaceful Return', key='C', scale='major', bpm=56, meter=4,
         bars_per_chord=2,
         prog=[('F', 'maj9'), ('C', 'add9'), ('A', 'm7'), ('G', 'sus4')],
         sections=[('block', 1, False, 0.5), ('arp_wave', 2, True, 0.7), ('arp_up', 1, True, 0.75),
                   ('sparse', 1, True, 0.55), ('block', 1, False, 0.4)],
         final=('C', 'add9')),
]


# ---------------------------------------------------------------- 작곡
def fit_length(spec, target=TARGET_SEC):
    """곡 길이를 target초에 맞춘다.

    섹션을 '코드 한 개 = 한 칸' 단위로 펼친 뒤, 도입과 종결부는 그대로 두고
    가운데(주제·전개)에서 뒤쪽 칸을 덜어낸다. 남는 1칸 미만의 오차는
    템포를 몇 % 이내로 미세 조정해 흡수하므로 곡마다의 템포 성격은 유지된다.
    """
    slots = []
    for kind, reps, mel, dyn in spec['sections']:
        for _ in range(reps):
            for ci in range(len(spec['prog'])):
                slots.append((kind, mel, dyn, ci))
    n_chord = len(spec['prog'])
    beat = 60.0 / spec['bpm']
    chord_beats = spec['bars_per_chord'] * spec['meter']
    ring_beats = spec['meter'] * 3
    avail = target - 4.0                       # 4초는 마지막 잔향 꼬리
    n = round((avail / beat - ring_beats) / chord_beats)
    n = max(2 * n_chord, min(len(slots), n))
    head, tail = slots[:n_chord], slots[-n_chord:]
    middle = slots[n_chord:-n_chord][: n - 2 * n_chord]
    slots = head + middle + tail
    bpm = (len(slots) * chord_beats + ring_beats) * 60.0 / avail
    return dict(spec, slots=slots, bpm=round(bpm, 2))


def chord_pitches(root, quality, base=48):
    r = NOTE[root]
    low = base + r if base + r <= base + 7 else base + r - 12
    return [low + iv for iv in QUALITY[quality]]


def scale_pitches(key, scale, lo, hi):
    k = NOTE[key]
    return [p for p in range(lo, hi + 1) if (p - k) % 12 in SCALES[scale]]


class Composer:
    def __init__(self, spec):
        self.s = spec
        self.rng = random.Random(spec['no'] * 7919)
        self.beat = 60.0 / spec['bpm']
        self.bar = self.beat * spec['meter']
        self.notes = []  # (start_sec, dur_sec, pitch, velocity 0..1)

    def add(self, t, dur, pitch, vel):
        j = self.rng.uniform(-0.012, 0.012)
        v = max(0.08, min(1.0, vel * self.rng.uniform(0.9, 1.08)))
        self.notes.append((max(0.0, t + j), dur, pitch, v))

    # --- 반주 텍스처 ------------------------------------------------------
    def bass(self, t, length, root, dyn):
        r = chord_pitches(root, '5', base=36)
        self.add(t, length, r[0], 0.55 * dyn)
        if self.rng.random() < 0.6:
            self.add(t + self.bar, length - self.bar, r[1], 0.35 * dyn)

    def texture(self, kind, t, length, root, qual, dyn):
        s, rng = self.s, self.rng
        ch = chord_pitches(root, qual, base=55)
        m = s['meter']
        if kind == 'block':
            for i, p in enumerate(ch):
                self.add(t + i * 0.06, length, p, 0.42 * dyn)
            self.add(t, length, chord_pitches(root, '5', 36)[0], 0.5 * dyn)
        elif kind == 'pedal':
            self.add(t, length, chord_pitches(s['key'], '5', 36)[0], 0.5 * dyn)
            self.add(t, length, chord_pitches(s['key'], '5', 36)[1], 0.35 * dyn)
            for b in range(0, int(length / self.beat), 2):
                p = rng.choice(ch)
                self.add(t + b * self.beat, self.beat * 3, p, 0.32 * dyn)
        elif kind in ('arp_up', 'arp_wave'):
            self.bass(t, length, root, dyn)
            pool = ch + [p + 12 for p in ch[:3]]
            if kind == 'arp_wave':
                pool = pool + pool[-2:0:-1]
            step = self.beat / 2
            n = int(round(length / step))
            for i in range(n):
                p = pool[i % len(pool)]
                accent = 1.0 if i % (m * 2) == 0 else 0.8
                self.add(t + i * step, step * 5, p, 0.36 * dyn * accent)
        elif kind == 'sparse':
            self.bass(t, length, root, dyn)
            hi = [p + 12 for p in ch]
            k = 0.0
            while k < length - self.beat:
                self.add(t + k, self.beat * 4, rng.choice(hi), 0.3 * dyn)
                k += self.beat * rng.choice([1.5, 2, 2, 3])
        elif kind == 'rain':
            self.bass(t, length, root, dyn)
            hi = [p + 12 for p in ch] + [p + 24 for p in ch[:2]]
            step = self.beat / 4
            for i in range(int(length / step)):
                if rng.random() < 0.28:
                    self.add(t + i * step, self.beat * 2, rng.choice(hi), rng.uniform(0.15, 0.3) * dyn)
        elif kind == 'waltz':
            bars = int(round(length / self.bar))
            for b in range(bars):
                tb = t + b * self.bar
                self.add(tb, self.bar * 1.5, chord_pitches(root, '5', 36)[0], 0.5 * dyn)
                self.add(tb + self.beat, self.beat * 2, ch[1], 0.3 * dyn)
                self.add(tb + self.beat, self.beat * 2, ch[2], 0.28 * dyn)
                self.add(tb + 2 * self.beat, self.beat * 2, ch[-1], 0.28 * dyn)

    # --- 멜로디 ----------------------------------------------------------
    def make_motif(self):
        """코드 하나당 한 개의 프레이즈(리듬+윤곽)를 만든다. 반복 시 재사용해 주제성을 준다."""
        rng, m = self.rng, self.s['meter']
        total = self.s['bars_per_chord'] * m
        rhythms4 = [[2, 1, 1, 4], [3, 1, 4], [1, 1, 2, 4], [4, 2, 2], [2, 2, 4], [1.5, 0.5, 2, 4]]
        rhythms3 = [[2, 1, 3], [3, 3], [1, 1, 1, 3], [2, 1, 2, 1]]
        phrases = []
        for _ in self.s['prog']:
            beats, lens = 0, []
            while beats < total:
                r = rng.choice(rhythms4 if m == 4 else rhythms3)
                for d in r:
                    if beats + d > total:
                        d = total - beats
                    if d > 0:
                        lens.append(d)
                        beats += d
            contour = [rng.choice([-2, -1, -1, 0, 1, 1, 2]) for _ in lens]
            rests = [rng.random() < 0.18 for _ in lens]
            rests[0] = rng.random() < 0.3
            phrases.append((lens, contour, rests))
        return phrases

    def melody(self, t, phrase, root, qual, dyn, vary):
        s, rng = self.s, self.rng
        scale = scale_pitches(s['key'], s['scale'], 67, 88)
        ch_pcs = {(p % 12) for p in chord_pitches(root, qual)}
        chord_tones = [p for p in range(67, 89) if p % 12 in ch_pcs]
        lens, contour, rests = phrase
        cur = getattr(self, '_mel', rng.choice(chord_tones))
        k = 0.0
        for i, (d, c, rest) in enumerate(zip(lens, contour, rests)):
            if vary and rng.random() < 0.25:
                c = rng.choice([-1, 0, 1])
            if i == 0 or d >= 3:  # 강박·긴 음은 코드톤으로 착지
                cur = min(chord_tones, key=lambda p: abs(p - cur) + rng.random())
            else:
                idx = min(range(len(scale)), key=lambda j: abs(scale[j] - cur))
                idx = max(0, min(len(scale) - 1, idx + c))
                cur = scale[idx]
            if not rest:
                self.add(t + k * self.beat, d * self.beat * 1.3, cur, 0.5 * dyn)
            k += d
        self._mel = cur

    # --- 전체 구조 --------------------------------------------------------
    def compose(self):
        s = self.s
        chord_len = s['bars_per_chord'] * self.bar
        motif = self.make_motif()
        t = 0.0
        played = set()
        for kind, mel, dyn, ci in s['slots']:
            root, qual = s['prog'][ci]
            self.texture(kind, t, chord_len, root, qual, dyn)
            if mel:
                self.melody(t, motif[ci], root, qual, dyn, vary=ci in played)
                played.add(ci)
            t += chord_len
        # 종지: 으뜸화음을 길게 울리고 끝낸다
        root, qual = s.get('final', (s['key'], 'add9' if s['scale'] in ('major', 'lydian', 'penta') else 'm9'))
        ring = self.bar * 3
        self.add(t, ring, chord_pitches(root, '5', 36)[0], 0.45)
        for i, p in enumerate(chord_pitches(root, qual, 55)):
            self.add(t + 0.25 + i * 0.12, ring, p, 0.3)
        self.add(t + 0.25 + 0.6, ring, chord_pitches(root, 'maj', 72)[0], 0.28)
        self.length = t + ring + 4.0
        self.notes.sort()
        return self.notes


# ---------------------------------------------------------------- 음원 합성
def piano_note(pitch, vel, dur):
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    ring = min(dur + 1.2, 9.0)
    n = int(ring * SR)
    t = np.arange(n) / SR
    B = 0.0002 * (1 + (pitch - 40) / 40)   # 현의 비조화성
    out = np.zeros(n)
    bright = 0.5 + 0.5 * vel
    base_decay = 0.9 + (pitch - 48) * 0.035
    for k in range(1, 11):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > SR * 0.45:
            break
        amp = (bright ** (k - 1)) / k ** 1.1
        decay = base_decay * (1 + 0.45 * (k - 1))
        # 이중 감쇠(빠른 초기 + 느린 잔향), 미세 디튜닝으로 세 현의 맥놀이 표현
        env = 0.65 * np.exp(-t * decay * 1.6) + 0.35 * np.exp(-t * decay * 0.35)
        det = 1 + 0.0007 * (k % 3 - 1)
        out += amp * env * (np.sin(2 * np.pi * fk * t) + 0.5 * np.sin(2 * np.pi * fk * det * t + k))
    atk = int(0.004 * SR)
    out[:atk] *= np.linspace(0, 1, atk)
    # 해머 타격음
    ham = np.random.default_rng(pitch).standard_normal(int(0.02 * SR)) * np.exp(-np.linspace(0, 8, int(0.02 * SR)))
    out[:len(ham)] += ham * 0.02 * vel
    # 건반에서 손을 뗀 뒤 댐퍼(부드러운 페달 느낌으로 길게)
    rel_start = int(min(dur, ring) * SR)
    if rel_start < n:
        out[rel_start:] *= np.exp(-np.arange(n - rel_start) / SR * 2.2)
    return out * vel * 0.2


def reverb_ir(seconds=4.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    sos = butter(2, 5000, 'low', fs=SR, output='sos')
    for c in range(2):
        noise = rng.standard_normal(n) * np.exp(-t * 1.5)
        ir[:, c] = sosfilt(sos, noise)
    ir[: int(0.02 * SR)] = 0  # pre-delay
    return ir / np.abs(ir).sum(axis=0) * 12


def render(notes, length):
    n = int(length * SR)
    dry = np.zeros((n, 2))
    cache = {}
    for start, dur, pitch, vel in notes:
        key = (pitch, round(vel, 2), round(dur, 1))
        if key not in cache:
            cache[key] = piano_note(pitch, vel, dur)
        x = cache[key]
        i = int(start * SR)
        j = min(n, i + len(x))
        pan = 0.5 + (pitch - 64) / 70
        dry[i:j, 0] += x[: j - i] * np.sqrt(1 - pan)
        dry[i:j, 1] += x[: j - i] * np.sqrt(pan)
    sos = butter(2, 7000, 'low', fs=SR, output='sos')
    dry = sosfilt(sos, dry, axis=0)
    ir = reverb_ir()
    wet = np.stack([fftconvolve(dry[:, c], ir[:, c])[:n] for c in range(2)], axis=1)
    mix = dry * 0.6 + wet * 0.55
    # 페이드 인/아웃
    fi, fo = int(1.0 * SR), int(5.0 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix *= 0.89 / np.max(np.abs(mix))
    return mix


def write_mp3(path, audio):
    pcm = (audio * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(SR)
    enc.set_channels(2)
    enc.set_quality(2)
    with open(path, 'wb') as f:
        f.write(enc.encode(pcm.tobytes()) + enc.flush())


def write_midi(path, spec, notes):
    mf = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    tr.append(mido.MetaMessage('track_name', name=f"{spec['no']:02d} {spec['en']}"))
    tr.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(spec['bpm'])))
    tr.append(mido.MetaMessage('time_signature', numerator=spec['meter'], denominator=4))
    tr.append(mido.Message('program_change', program=0))
    tr.append(mido.Message('control_change', control=64, value=100))  # 서스테인 페달
    tpb = 480 * spec['bpm'] / 60.0
    ev = []
    for s, d, p, v in notes:
        ev.append((int(s * tpb), 1, p, int(20 + v * 90)))
        ev.append((int((s + d) * tpb), 0, p, 0))
    ev.sort()
    last = 0
    for tick, on, p, v in ev:
        tr.append(mido.Message('note_on' if on else 'note_off', note=p, velocity=v, time=tick - last))
        last = tick
    mf.save(path)


def main(selected):
    for spec in TRACKS:
        if selected and spec['no'] not in selected:
            continue
        spec = fit_length(spec)
        c = Composer(spec)
        notes = c.compose()
        base = f"{spec['no']:02d}_{spec['en'].replace(' ', '_')}"
        write_midi(os.path.join(OUT, 'midi', base + '.mid'), spec, notes)
        write_mp3(os.path.join(OUT, 'audio', base + '.mp3'), render(notes, c.length))
        m, s = divmod(int(c.length), 60)
        print(f"{spec['no']:02d} {spec['title']} ({spec['en']}) {m}:{s:02d}  bpm={spec['bpm']}  notes={len(notes)}")


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT, 'midi'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'audio'), exist_ok=True)
    main({int(a) for a in sys.argv[1:]})
