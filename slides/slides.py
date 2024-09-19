from fri import *
import shutil
from pprint import pp
from time import time
from typing import Type
from manim import *
from manim.utils.file_ops import open_file as open_media_file
from merkle import MerkleTree
from polynomial import X, Polynomial, interpolate_poly, prod
from field import FieldElement as F
from pathlib import Path
import scipy
media_path = Path("media")
media_path.mkdir(exist_ok=True, parents=True)

def to_shortest_repr(val):
    return (val + F.k_modulus // 2) % F.k_modulus - F.k_modulus // 2

data = [F(1), F(1)]
while len(data) < 8:
    data.append(data[-1] + data[-2])
g = F.generator() ** 2
assert g.is_order(8)
G = [g ** i for i in range(8)]
poly = interpolate_poly(G, data)
int_data = [d.val for d in data]

data = [1, 1]
while len(data) < 8:
    data.append((data[-1] + data[-2]) % 17)
_fake_poly = scipy.interpolate.lagrange(range(8), data)
fake_poly = lambda x: _fake_poly(x) % 17

def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data

#def repr_latex(poly, name='', var='x', mod=0, markers='$$'):
#    if isinstance(poly, Polynomial):
#        poly = poly.poly
#    var = "(%s)" % var if var else ""
#    prefix = "%s%s = " % (name, var) if name else ""
#    prefix = markers[0] + prefix
#    suffix = " \\text{ mod } %s" % mod if mod else ""
#    suffix = suffix + markers[1]
#    return tuple(flatten((prefix, *(
#        (str(c), "x^{%s}" % i if i > 1
#        else "x" if i == 1
#        else "")
#        for i, c in enumerate(poly)
#        if str(c) != '0'), suffix)))
#

class IntroductionSlide(Scene):
    def construct(self):
        self.next_section("Introduction")
        last = None
        bg = Rectangle(width=15, height=3, color="#35B1B0", fill_opacity=1).to_edge(UP)
        title1 = Tex(r"Achieving Transparent Succinctness", font_size=56).next_to(bg, DOWN).to_edge(LEFT)
        title2 = Tex("for provable computation", font_size=56).next_to(title1, DOWN).align_to(title1, LEFT)
        title = VGroup(title1, title2)
        bar = Line(LEFT * 7, LEFT * 6, color=WHITE).next_to(title, DOWN).align_to(title, LEFT).shift(DOWN/2)
        by = Tex(r"FRI - by Thybault Alabarbe, 2024", color=GREY).scale(.9).next_to(bar, DOWN).align_to(bar, LEFT).shift(DOWN/2)
        self.add(bg)
        self.play(Create(title), Create(bar), Create(by))
        self.wait(4)


class StarksBirdsEyeViewSlide(Scene):
    def construct(self):
        self.next_section("StarksBirdsEyeView")
        title = Tex("Starks - a bird's eye view", font_size=96)
        self.play(Create(title))
        self.wait(2)

def repr_latex(poly, name='', var='x', split_var=False):
    if isinstance(poly, Polynomial):
        poly = poly.poly
    if not name:
        var = ""
    if var:
        if split_var:
            var = ("(", var, ")")
            prefix = (name, *var, "=")
        else:
            var = "(%s)" % var 
            prefix = (name + var, "=")
    else:
        prefix = ()
    res = []
    first = True
    for i, c in enumerate(poly):
        s = str(c)
        if s != '0':
            if s.startswith("-"):
                sign = "-"
            else:
                sign = "+"
            if sign == "+":
                if not first:
                    res.append(sign)
            if i == 0:
                res.append(s)
            elif i == 1:
                res.extend((s, "x"))
            else:
                res.extend((s, "x^{%s}" % i))
            first = False
    return *prefix, *res

justified_tex_template = TexTemplate(
    documentclass=r"\documentclass[preview]{standalone}"
)
justified_tex_template.add_to_preamble(r"\usepackage{ragged2e}")
poly_latex = repr_latex(poly, name='f')
poly_tex = MathTex(*poly_latex)

class GeneratorSlide(Scene):
    def construct(self):
        self.next_section("generator")
        title = Tex("Detour through field generators", font_size=56).to_edge(UP)
        self.play(Create(title))
        self.wait(1)
        g = F.generator()
        assert g.is_order(16)
        self.wait(1)
        g_tex = MathTex(r"g = %s" % g)
        self.play(Write(g_tex))
        circle = Circle(radius=2.5, color=WHITE, stroke_width=2)
        self.play(Create(circle))
        labels = VGroup()
        for i in range(17):
            label = LabeledDot(MathTex(r"%s" % i), color=BLACK)
            label.next_to(circle.point_at_angle(i * TAU / 17), OUT)
            labels.add(label)
        self.play(Create(labels))
        self.wait(1)
        while g != F(1):
            val = F(1)
            first = True
            prev = val
            lines = []
            new_labels = labels.copy()
            while val != F(1) or first:
                lines.append(Line(labels[prev.val], labels[val.val], color=RED))
                self.play(Create(lines[-1]), run_time=.2)
                new_labels[val.val] = LabeledDot(MathTex(r"%s" % val.val), color=RED).next_to(circle.point_at_angle(val.val * TAU / 17), OUT)
                self.play(TransformMatchingShapes(labels[val.val], new_labels[val.val]), run_time=.1)
                prev = val
                val = val * g
                first = False
            lines.append(Line(labels[prev.val], labels[val.val], color=RED))
            self.play(Create(lines[-1]))
            self.wait(1)
            self.play(FadeOut(*lines))
            self.play(*(
                    TransformMatchingShapes(new_label, label)
                 for new_label, label in zip(new_labels, labels)))
            g = g**2
            if g == F(1):
                break
            g_tex_new = MathTex(r"g = %s" % g.val)
            self.play(TransformMatchingTex(g_tex, g_tex_new))
            g_tex = g_tex_new
        self.wait(1)


class InterpolationSlide(Scene):
    def construct(self):
        self.next_section("introduction")
        title = Tex("Starks - Execution Trace", font_size=56).to_edge(UP)
        code = Code(
            code="""\
            def fib(a, b):
                for _ in range(2, 8):
                    a, b = b, (a + b) % 17 
            """,
            language="python",
            tab_width=4,
            background="window",
        ).next_to(title, DOWN)
        self.play(Create(title))
        self.play(Create(code))
        self.wait(1)
        values = [str(d) for d in int_data]
        col_labels = [Tex(r"\textbf{$f(\omega^{%s})$}" % i) for i in range(len(int_data))]
        table = Table([values], col_labels=col_labels, include_outer_lines=True).scale(0.5).next_to(code, DOWN)
        self.play(Create(table))
        self.wait(5)

        self.next_section("interpolation")
        last_title = title
        title = Tex("Starks - Interpolation", font_size=56).to_edge(UP)
        self.play(TransformMatchingTex(last_title, title))
        self.wait(1)
        self.play(code.animate.scale(.8).to_edge(UR).shift(DOWN),
                  table.animate.scale(.7).to_edge(UL).shift(DOWN * 1.2))
        self.wait(1)
        ax = Axes(
            x_range=[0, 7.5, 1],
            y_range=[0, 17, 2],
            y_axis_config={"include_numbers": True},
        ).scale(0.6).to_edge(DL).shift(UP)
        for i in range(len(int_data)):
            label = Tex(r"$\omega^{%s}$" % i).scale(.5)
            label.next_to(ax.coords_to_point(i, 0), DOWN)
            ax.x_axis.add(label)
        self.play(Create(ax))
        dots = VGroup()
        for i, x in enumerate(data):
            dots.add(Dot(ax.coords_to_point(i, x), color=WHITE))
        self.play(Create(dots))
        self.wait(1)
        comment = VGroup()
        comment.add(poly_tex.next_to(ax, DOWN).align_to(ax, LEFT))
        self.play(Create(comment))
        self.wait(1)
        self.play(Create(ax.plot(fake_poly, color=BLUE)))
        comment.add(Tex(r"\textit{This is \textbf{NOT} the Field polynomial!}", font_size=32, color=YELLOW).scale(.8).next_to(ax, RIGHT))
        comment.add(Tex(r"\textit{Just a useful visualization in $\mathbb{R}$ mod $17$}", font_size=32, color=YELLOW).scale(.8).next_to(comment[-1], DOWN/2).align_to(comment[-1], LEFT))
        self.wait(5)

        self.next_section("low_degree_extension")
        last_title = title
        title = Tex("Starks - Low Degree Extension", font_size=56).to_edge(UP)
        self.play(TransformMatchingTex(last_title, title))
        self.wait(1)
        blowup = 2
        extended_dot = VGroup()
        ax2 = Axes(
            x_range=[0, 7.5, .5],
            y_range=[0, 17, 2],
            y_axis_config={"include_numbers": True},
        ).scale(0.6).to_edge(DL).shift(UP)
        self.replace(ax, ax2)
        for i in range(len(data) * blowup):
            x = i / blowup
            extended_dot.add(Dot(ax2.coords_to_point(x, fake_poly(x)), color=RED).scale(.7))
            label = Tex(r"$g^{%s}$" % i).scale(.5)
            label.next_to(ax2.coords_to_point(x, 0), DOWN)
            ax2.x_axis.add(label)
        self.play(Create(extended_dot))
        self.wait(5)

p0 = (poly - 1) / (X - g ** 0)
p1 = (poly - 4) / (X - g ** 15)
p2 = (poly(g ** 2 * X) - poly(g * X) - poly(X)) / prod([X - g ** i for i in range(len(G) - 3)])

class ConstraintsSlide(Scene):
    def construct(self):
        global p0, p1, p2
        self.next_section("constraints")
        title = Tex("Starks - Constraints", font_size=56).to_edge(UP)
        last_title = title
        self.play(Create(title))
        self.wait(1)
        self.add(poly_tex.to_edge(DOWN))
        self.wait(3)
        self.play(poly_tex.animate.next_to(title, DOWN))
        comment = VGroup()
        comment.add(Tex(r"\textbf{Constraints:}").next_to(poly_tex, DOWN).align_to(poly_tex, LEFT))
        self.play(Create(comment))
        def render_solve(*steps, align_to=comment):
            nonlocal title, last_title
            sections = [s[0] for s in steps]
            steps = [s[1] for s in steps]
            prev = comment
            systems = []
            for s, system in enumerate(steps):
                if sections[s]:
                    self.next_section(sections[s])
                    self.play(TransformMatchingTex(last_title, title))
                    last_title = title
                    self.wait(1)
                systems.append([])
                if any(r"\frac" in sym for eq in system for sym in eq):
                    pp(systems)
                    transforms = []
                    for p, preveq in enumerate(systems[-2]):
                        new_eq = preveq.copy()
                        new_eq.move_to(preveq.get_center()
                            + DOWN * (p * .6 + .25))
                        transforms.append(TransformMatchingTex(preveq, new_eq))
                        systems[-2][p] = new_eq
                    self.play(*transforms)
                for e, eq in enumerate(system):
                    if e == 0:
                        if systems[-1]:
                            prev = systems[-1][0]
                        else:
                            prev = comment
                        text = MathTex(*eq).next_to(align_to, DOWN)
                    else:
                        text = MathTex(*eq).next_to(prev, DOWN)
                    text = text.align_to(prev, LEFT)
                    if s == 0:
                        self.play(Create(text))
                    else:
                        self.play(TransformMatchingTex(systems[-2][e], text))
                    systems[-1].append(text)
                    prev = text
                self.wait(1)
        render_solve(
            (None, [
                ["f", "(", "g^{0}", ")", "=", "1"],
                ["f", "(", "g^{15}", ")", "=", "4"],
                ["f", "(", "g^{2}x", ")", "=", "f(gx)", "+", "f(x)"]
            ]),
            (None, [
                ["f", "(", "g^{0}", ")", "-", "1", "=", "0"],
                ["f", "(", "g^{15}", ")", "-", "4", "=", "0"],
                ["f", "(", "g^{2}x", ")", "-", "f", "(", "gx", ")", "-", "f", "(", "x", ")", "=", "0"]
            ]),
            ("composition_polynomial", [
                ["p_0", "(", "x", ")", "=", r"\frac{f(x) - 1}{x - g^{0}}"],
                ["p_1", "(", "x", ")", "=", r"\frac{f(x) - 4}{x - g^{15}}"],
                ["p_2", "(", "x", ")", "=", r"\frac{f(g^2x) - f(gx) - f(x)}{\prod \limits_{i = 0}^{8} {(x - g^{i})}}"]
            ]),
            (None, [
                repr_latex(p0, name='p_0', split_var=True),
                repr_latex(p1, name='p_1', split_var=True),
                repr_latex(p2, name='p_2', split_var=True)
            ]))
        cp = Tex(r"$CP(x) = \alpha_0 p_0(x) + \alpha_1 p_1(x) + \alpha_2 p_2(x)$").to_edge(DOWN)
        self.play(Create(cp))
        self.wait(5)

def short_hash(hash, name='', **kwargs):
    group = VGroup()
    group.add(Text(name+':' if name else name, **kwargs))
    group.add(MathTex(fr"0x{hash[:4]}...{hash[-4:]}", **kwargs).next_to(group[0], RIGHT * 2).scale(1.3))
    return group

def bubble(text, from_direction, **kwargs):
    text = Tex(text, **kwargs)
    ul = text.get_corner(UL) * np.array((1.2, 1.7, 1))
    ur = text.get_corner(UR) * np.array((1.2, 1.7, 1))
    dl = text.get_corner(DL) * np.array((1.2, 1.7, 1))
    dr = text.get_corner(DR) * np.array((1.2, 1.7, 1))
    ml = (ul + dl) / 2
    mr = (ur + dr) / 2
    offset = .5
    shape = Polygon(ul,
                    ml,
                    dl + offset * LEFT if from_direction is LEFT else dl,
                    dr + offset * RIGHT if from_direction is RIGHT else dr,
                    mr,
                    ur,
                    stroke_width=3, color=WHITE)
    return VGroup(shape, text).shift(DOWN / 2)

class FriIntroductionSlide(Scene):
    def construct(self):
        self.next_section("FRI_introduction")
        title = Tex("FRI", font_size=56).to_edge(UP)
        self.play(Create(title))
        self.wait(1)
        t2c = {'F': RED, 'R': RED, 'I': RED}
        first = Text("Stands for Fast Reed-Solomon Interactive Oracle Proofs of Proximity", t2c=t2c, font_size=24).next_to(title, DOWN).to_edge(LEFT)
        self.play(Create(first))
        second = Text("(Fast RS IOPP)… Let’s unpack:", t2c=t2c,
                      font_size=24).next_to(first, DOWN).to_edge(LEFT)
        self.play(Create(second))
        last_text = second
        self.wait(3)
        texts = [
            ('- "Fast": algorithm similar to the Fast Fourier Transform in structure and', 0),
            ('algorithmic complexity, provides succinctness', 5),
            ('- "Reed Solomon": error correcting codes using polynomials', 5),
            ('- "Interactive": protocol between a prover and a verifier', 5),
            ('- "Oracle": commitment via a Merkle tree', 5),
            ('- "Proofs": proof that we know a polynomial that satisfies a set of constraints', 5),
            ('- "of Proximity": because FRI shows that a function is close to a polynomial', 0),
            ('of low degree.', 5),
        ]
        for text, delay in texts:
            tex = Text(text, font_size=24).next_to(last_text, DOWN).to_edge(LEFT)
            self.play(Create(tex))
            last_text = tex
            self.wait(delay)


class FriSlide(Scene):
    def construct(self):
        self.next_section("FRI_composition_polynomial")
        title = Tex("FRI - Composition Polynomial", font_size=56).to_edge(UP)
        last_title = title
        self.play(Create(title))
        self.wait(1)
        g = F.generator()
        cp0 = MathTex(r"CP(x)", r"=", r"\alpha_0", r"p_0(x)", r"+", r"\alpha_1", r"p_1(x)", r"+", r"\alpha_2", r"p_2(x)").to_edge(DOWN)
        self.add(cp0)
        self.play(cp0.animate.to_edge(UP).shift(DOWN))
        prover = VGroup(SVGMobject("prover.svg").scale(.8).to_edge(LEFT).shift(DOWN/2))
        prover.add(Tex(r"\textbf{Prover}").next_to(prover[-1], DOWN))
        verifier = VGroup(SVGMobject("verifier.svg").scale(.8).to_edge(RIGHT).shift(DOWN/2))
        verifier.add(Tex(r"\textbf{Verifier}").next_to(verifier[-1], DOWN))
        self.play(Create(prover), Create(verifier))
        self.wait(1)
        bubble_prover = bubble(r"I need $\alpha_0$, $\alpha_1$, $\alpha_2$", from_direction=LEFT)
        self.play(Create(bubble_prover))
        self.wait(1)
        self.play(Uncreate(bubble_prover))
        alphas = (5, 8, 3)
        bubble_verifier = bubble(r"Use ($%s$, $%s$, $%s$)" % alphas, from_direction=RIGHT)
        self.play(Create(bubble_verifier))
        self.wait(1)
        cp1 = MathTex(r"CP_{0}(x)", r"=", str(alphas[0]), r"p_0(x)", r"+", str(alphas[1]), r"p_1(x)", r"+", str(alphas[2]), r"p_2(x)").to_edge(UP).shift(DOWN)
        self.play(TransformMatchingTex(cp0, cp1))
        self.wait(1)
        self.play(Uncreate(bubble_verifier))
        cp: Polynomial = sum(a * p for a, p in zip(alphas, (p0, p1, p2)))
        cp_tex = MathTex(*repr_latex(cp, name='CP_0')).to_edge(UP).shift(DOWN)
        self.play(TransformMatchingTex(cp1,  cp_tex))

        domain = [g ** i for i in range(16)]
        print(f"{domain=}")
        layer = [cp(d) for d in domain]
        layer_str = [str(d) for d in layer]
        col_labels = [Tex(r"\textbf{$CP_0(g^{%s})$ }" % i) for i in range(len(layer))]
        table = Table([layer_str], col_labels=col_labels, include_outer_lines=True).scale(0.5).next_to((0, 0, 0), RIGHT).to_edge(DOWN)
        self.play(Create(table))
        table.generate_target()
        table.target.align_to(prover, RIGHT).to_edge(DOWN)
        self.play(MoveToTarget(table, rate_func=rate_functions.ease_in_sine, run_time=3))
        self.wait(1)
        merkle = MerkleTree(layer_str)
        merkle_root = short_hash(merkle.root, name="Oracle CP_0 merkle root", color=GREEN).next_to(table, RIGHT).scale(.5)
        self.play(Create(merkle_root))
        self.wait(1)
        merkle_root.generate_target()
        merkle_root.target.next_to(verifier, UP).scale(.7).to_edge(RIGHT)
        self.play(MoveToTarget(merkle_root, rate_func=rate_functions.ease_in_sine, run_time=1))

        cp_tex2 = cp_tex.copy().to_edge(UP).shift(DOWN)
        self.add(cp_tex2)
        last_cp_tex = cp_tex.copy().to_edge(UP).shift(DOWN)
        self.add(last_cp_tex)
        last_cp = cp
        self.wait(1)
        betas = (2, 8, 11)
        last_even_tex = cp_tex
        last_odd_tex = cp_tex2
        fri_domains = [domain]
        fri_layers = [layer]
        fri_polys = [cp]
        fri_merkles = [merkle]
        g = g ** 2
        query = 7
        self.play(Uncreate(table))

        def render_solve(*steps, align_to=None, scale=1.0, to_edge=None, shift=None):
            nonlocal title, last_title
            print("Rendering solve")
            pp(steps)
            if align_to is None:
                align_to = prover
            sections = [s[0] for s in steps]
            steps = [s[1] for s in steps]
            prev = align_to
            systems = []
            for s, system in enumerate(steps):
                print(f"{s=}")
                if sections[s]:
                    self.next_section(sections[s])
                    self.play(TransformMatchingTex(last_title, title))
                    last_title = title
                systems.append([])
                if any(r"\frac" in sym for eq in system for sym in eq):
                    pp(systems)
                    transforms = []
                    for p, preveq in enumerate(systems[-2]):
                        new_eq = preveq.copy()
                        new_eq.move_to(preveq.get_center()
                            + DOWN * (p * .6 + .25))
                        transforms.append(TransformMatchingTex(preveq, new_eq))
                        systems[-2][p] = new_eq
                    self.play(*transforms)
                for e, eq in enumerate(system):
                    print(f"{e=}")
                    if e == 0:
                        if systems[-1]:
                            prev = systems[-1][0]
                            text = MathTex(*eq).next_to(prev, DOWN)
                        else:
                            prev = align_to
                            text = MathTex(*eq).next_to(align_to, DOWN)
                        print(f"{text=}")
                    else:
                        text = MathTex(*eq).next_to(prev, DOWN)
                        print(f"{text=}")
                    text = text.align_to(prev, LEFT)
                    print(f"{text=}")
                    if s == 0 and e == 0:
                        text = text.shift(RIGHT * 2)
                    else:
                        text.align_to(align_to, LEFT).shift(RIGHT * 1.8)
                    if s > 0:
                        text = text.shift(RIGHT * 1)
                    text = text.scale(scale)
                    if to_edge is not None:
                        text = text.to_edge(to_edge).shift(LEFT * 3.1)
                    if shift is not None and e == 0:
                        text = text.shift(shift)
                    if s == 0:
                        self.play(Create(text))
                    else:
                        self.play(TransformMatchingTex(systems[-2][e], text))
                    systems[-1].append(text)
                    prev = text
                self.wait(1)
            return VGroup(*systems[-1])
        for i in range(1, 4):
            self.wait(1)
            self.next_section("FRI_steps_%s_commit" % i)
            title = Tex("FRI - Step %s: Commit" % (i+1), font_size=56).to_edge(UP)
            self.play(TransformMatchingTex(last_title, title))
            self.wait(1)
            last_title = title
            even_cp = Polynomial([last_cp.poly[i] if i % 2 == 0 else 0 for i in range(len(last_cp.poly))])
            odd_cp = Polynomial([last_cp.poly[i] if i % 2 == 1 else 0 for i in range(len(last_cp.poly))])

            even_tex = MathTex(*repr_latex(even_cp, name='CP_{%s even}' % i, var='x^2')).next_to(last_cp_tex, DOWN).align_to(last_cp_tex, LEFT)
            odd_tex = MathTex(*repr_latex(odd_cp, name='xCP_{%s odd}' % i, var='x^2')).next_to(even_tex, DOWN).align_to(even_tex, LEFT)

            self.play(TransformMatchingTex(last_even_tex, even_tex))
            last_even_tex = even_tex
            self.wait(1)
            self.play(TransformMatchingTex(last_odd_tex, odd_tex))
            last_odd_tex = odd_tex
            self.wait(1)

            real_even_cp = Polynomial([even_cp.poly[x] for x in range(len(even_cp.poly)) if x % 2 == 0])
            real_odd_cp = Polynomial([odd_cp.poly[x] for x in range(len(odd_cp.poly)) if x % 2 == 1])

            real_even_tex = MathTex(*repr_latex(real_even_cp, name='CP_{%s even}' % i)).next_to(last_cp_tex, DOWN).align_to(last_cp_tex, LEFT)
            real_odd_tex = MathTex(*repr_latex(real_odd_cp, name='CP_{%s odd}' % i)).next_to(real_even_tex, DOWN).align_to(real_even_tex, LEFT)

            self.play(TransformMatchingTex(last_even_tex, real_even_tex))
            last_even_tex = real_even_tex
            self.wait(1)
            self.play(TransformMatchingTex(last_odd_tex, real_odd_tex))
            last_odd_tex = real_odd_tex
            self.wait(1)

            cp_unexpanded = MathTex("CP_{%s}" % i, r"=", *repr_latex(real_even_cp), "+", r"\beta_%i" % (i-1), "(", *repr_latex(real_odd_cp), ")").to_edge(UP).align_to(last_cp_tex, LEFT).shift(DOWN)
            self.play(TransformMatchingTex(last_cp_tex, cp_unexpanded))
            last_cp_tex = cp_unexpanded
            self.wait(1)

            bubble_prover = bubble(r"I need $\beta_%s$" % (i - 1), from_direction=LEFT)
            self.play(Create(bubble_prover))
            self.wait(1)
            beta = betas[i - 1]
            bubble_verifier = bubble(r"$\beta_%s = %s$" % (i - 1, beta), from_direction=RIGHT)
            self.play(Transform(bubble_prover, bubble_verifier))
            self.wait(1)

            copy_bubble = bubble_prover.copy()
            beta_tex = copy_bubble[1]
            self.add(beta_tex)
            self.remove(copy_bubble)
            beta_tex.generate_target()
            beta_tex.target.next_to(verifier, DOWN).scale(.5)
            self.play(MoveToTarget(beta_tex, rate_func=rate_functions.ease_in_sine, run_time=1))
            self.wait(1)

            verifying_values = VGroup()
            verifying_values.add(beta_tex.copy())

            cp_unexpanded = MathTex("CP_{%s}" % i, r"=", *repr_latex(real_even_cp), "+", beta, "(",  *repr_latex(real_odd_cp), ")").to_edge(UP).align_to(last_cp_tex, LEFT).shift(DOWN)
            self.play(TransformMatchingTex(last_cp_tex, cp_unexpanded))
            last_cp_tex = cp_unexpanded
            self.wait(1)

            cp_next = real_even_cp + beta * real_odd_cp
            fri_polys.append(cp_next)
            cp_next_tex = MathTex(*repr_latex(cp_next, name='CP_%s' % (i))).to_edge(UP).align_to(last_cp_tex, LEFT).shift(DOWN)
            self.play(Uncreate(bubble_prover), Uncreate(bubble_verifier))
            self.play(TransformMatchingTex(last_cp_tex, cp_next_tex))
            last_cp_tex = cp_next_tex
            last_cp = cp_next
            self.wait(1)

            g = g ** 2
            domain = next_fri_domain(domain)
            print(f"{domain=}")
            fri_domains.append(domain)
            layer = [cp_next(d) for d in domain]
            layer_str = [str(d) for d in layer]
            print(f"{layer=}")
            fri_layers.append(layer)
            col_labels = [Tex(r"\textbf{$CP_0(g^{%s})$ }" % l) for l in range(len(layer))]
            table = Table([layer_str], col_labels=col_labels, include_outer_lines=True).scale(0.5).to_edge(DOWN)
            self.play(Create(table))
            self.wait(1)
            table.generate_target()
            table.target.align_to(prover, RIGHT).to_edge(DOWN)
            self.play(MoveToTarget(table, rate_func=rate_functions.ease_in_sine, run_time=1))
            merkle = MerkleTree(layer_str)
            fri_merkles.append(merkle)
            merkle_root = short_hash(merkle.root, name=f"Oracle CP_{i} merkle root", color=GREEN).next_to(table, RIGHT).scale(.5)
            self.play(Create(merkle_root))
            self.wait(1)
            prover_bubble = bubble(text="Here is the merkle root", from_direction=LEFT)
            self.play(Create(prover_bubble))
            self.wait(1)
            self.play(Uncreate(prover_bubble))
            merkle_root.generate_target()
            merkle_root.target.next_to(verifier, UP).shift(UP * i/3).scale(.7).to_edge(RIGHT)
            self.play(MoveToTarget(merkle_root, rate_func=rate_functions.ease_in_sine, run_time=1))
            self.play(Uncreate(table))
            self.wait(1)

            self.next_section("FRI_steps_%s_decommit" % i)
            title = Tex("FRI - Step %s: Decommit" % i, font_size=56).to_edge(UP)
            self.play(TransformMatchingTex(last_title, title))
            last_title = title
            self.wait(1)
            verifier_bubble = bubble(text="Let's check...", from_direction=RIGHT)
            self.play(Create(verifier_bubble))
            self.wait(1)
            old_bubble = verifier_bubble
            verifier_bubble = bubble(text="What's at index $%s$?" % query, from_direction=RIGHT)
            self.play(Transform(old_bubble, verifier_bubble))
            self.wait(1)
            self.play(Uncreate(old_bubble))
            self.wait(1)
            length = len(domain)
            idx = query
            idx = idx % length
            sib_idx = (idx + length // 2) % length        
            print(f"{idx=}")
            print(f"{domain[idx]=}")
            print(f"{layer[idx]=}")
            print(f"{sib_idx=}")
            print(f"{domain[sib_idx]=}")
            print(f"{layer[sib_idx]=}")

            # layer[idx]
            prover_bubble = bubble(text="$CP_{%s}(%s) = %s$" % (i, domain[idx], layer[idx]), from_direction=LEFT)
            self.play(Create(prover_bubble))
            last_bubble = prover_bubble
            self.wait(1)

            copy_bubble = prover_bubble.copy()
            layer_idx = copy_bubble[1]
            self.add(layer_idx)
            self.remove(copy_bubble)
            layer_idx.generate_target()
            layer_idx.target.next_to(verifier, DOWN).shift(DOWN * 1/3).scale(.5)
            self.play(MoveToTarget(layer_idx, rate_func=rate_functions.ease_in_sine, run_time=1))
            self.wait(1)
            verifying_values.add(layer_idx.copy())

            prover_bubble = bubble(text="$Proof_{%s}(%s)$: ..." % (i, domain[idx]), from_direction=LEFT) 
            self.play(Transform(last_bubble, prover_bubble))

            copy_bubble = prover_bubble.copy()
            layer_idx_proof = copy_bubble[1]
            self.add(layer_idx_proof)
            self.remove(copy_bubble)
            layer_idx_proof.generate_target()
            layer_idx_proof.target.next_to(verifier, DOWN).shift(DOWN * 2/3).scale(.5)
            self.play(MoveToTarget(layer_idx_proof, rate_func=rate_functions.ease_in_sine, run_time=1))
            self.wait(1)
            verifying_values.add(layer_idx_proof.copy())

            # layer[sib_idx]
            prover_bubble = bubble(text="$CP_{%s}(%s) = %s$" % (i, domain[sib_idx], layer[sib_idx]), from_direction=LEFT)
            self.play(Transform(last_bubble, prover_bubble))
            self.wait(1)

            copy_bubble = prover_bubble.copy()
            layer_sib_idx = copy_bubble[1]
            self.add(layer_sib_idx)
            layer_sib_idx.generate_target()
            layer_sib_idx.target.next_to(verifier, DOWN).shift(DOWN * 3/3).scale(.5)
            self.play(MoveToTarget(layer_sib_idx, rate_func=rate_functions.ease_in_sine, run_time=1))
            self.wait(1)
            verifying_values.add(layer_sib_idx.copy())
            self.remove(copy_bubble)

            prover_bubble = bubble(text="$Proof_{%s}(%s)$: ..." % (i, domain[sib_idx]), from_direction=LEFT) 
            self.play(Transform(last_bubble, prover_bubble))

            copy_bubble = prover_bubble.copy()
            layer_sib_idx_proof = copy_bubble[1]
            self.add(layer_sib_idx_proof)
            self.remove(copy_bubble)
            layer_sib_idx_proof.generate_target()
            layer_sib_idx_proof.target.next_to(verifier, DOWN).shift(DOWN * 4/3).scale(.5)
            self.play(MoveToTarget(layer_sib_idx_proof, rate_func=rate_functions.ease_in_sine, run_time=1))
            self.wait(1)
            verifying_values.add(layer_sib_idx_proof.copy())

            self.play(Uncreate(last_bubble))
            self.wait(1)

            self.next_section("FRI_steps_%s_verify" % i)
            title = Tex("FRI - Step %s: Verify" % i, font_size=56).to_edge(UP)
            self.play(TransformMatchingTex(last_title, title))
            last_title = title
            self.wait(1)
            # check that foldings are correct
            # namely, that the answer is the correct folding of the previous layer
            # with the beta value evaluated at recorded element
            # let p be the previous layer, and f be the current layer
            # f(beta) = p0(beta²) + beta * p1(beta²)
            # f(-beta) = p0(beta²) - beta * p1(beta²)
            # f0 = f(beta) is the answer at the current layer
            # f1 = f(-beta) is the sibling answer at the current layer
            # p0 = p(beta²) is the answer at the previous layer
            # p1 = p(beta²) is the sibling answer at the previous layer
            # so we have:
            # f0 = p0 + beta * p1
            # f1 = p0 - beta * p1
            # we solve for p0 and p1:
            # p0 = (f0 + f1) / 2
            # p1 = (f0 - f1) / (2 * beta)
            # we check that p0 and p1 are the correct answers at the previous layer
            # which means:
            # f0 = p0 + p1 * beta
            # and that the merkle proofs are correct
            CP_idx = layer[idx]
            print(f"{CP_idx=}")
            CP_sib_idx = layer[sib_idx]
            print(f"{CP_sib_idx=}")
            prev_idx = (CP_idx + CP_sib_idx) / F(2)
            print(f"{prev_idx=}")
            prev_sib_idx = (CP_idx - CP_sib_idx) / (F(2) * F(beta))
            print(f"{prev_sib_idx=}")
            res = prev_idx + prev_sib_idx * beta
            print(f"{res=}")
            print(f"{type(CP_idx)=}")
            print(f"{type(CP_sib_idx)=}")
            last_system = render_solve(
                (None, [
                    ["CP_{%s}(%s)" % (i - 1, domain[idx]),
                     "=", "(", str(CP_idx),
                     "+", f"({CP_sib_idx})" if CP_sib_idx.val > F.k_modulus // 2 else str(CP_sib_idx),
                     ")", r"\div", "2",
                     ],
                    ["CP_{%s}(%s)" % (i - 1, domain[sib_idx]),
                     "=",
                     "(", str(CP_idx),
                     "-", f"({CP_sib_idx})" if CP_sib_idx.val > F.k_modulus // 2 else str(CP_sib_idx),
                     ")", r"\div", "(", "2", r"\cdot", str(beta), ")"
                     ]
                ]),
                (None, [
                    ["CP_{%s}(%s)" % (i - 1, domain[idx]),
                     "=", str(prev_idx)],
                    ["CP_{%s}(%s)" % (i - 1, domain[sib_idx]),
                     "=", str(prev_sib_idx)]
                ]),
                scale=.7, shift=UP/2)
            assert res == CP_idx, "The proof is incorrect"
            self.wait(1)
            result = render_solve(
                    (None, [[
                    r"CP_{%s}(%s)" % (i, domain[idx]),
                    "=", r"CP_{%s}(%s)" % (i - 1, domain[idx]),
                    "+", r"\beta_%s" % (i - 1),
                    "CP_{%s}(%s)" % (i - 1, domain[sib_idx])
                ]]),
                    (None, [[
                    r"CP_{%s}(%s)" % (i, domain[idx]),
                    "=", str(prev_idx),
                    "+", str(beta), r"\cdot", 
                    f"({prev_sib_idx})" if prev_sib_idx.val > F.k_modulus // 2 else str(prev_sib_idx)]]),
                    (None, [[
                    r"CP_{%s}(%s)" % (i, domain[idx]),
                    "=", str(res)]]),
                    scale=.7, align_to=last_system[-1], to_edge=DOWN)

            verifier_bubble = bubble(text="Yes it matches!", from_direction=RIGHT)
            self.play(Create(verifier_bubble))
            self.wait(1)
            self.play(Uncreate(verifier_bubble))
            self.play(Uncreate(result))
            self.play(Uncreate(verifying_values),
                      Uncreate(last_system),
                      *(Uncreate(obj) 
                      for obj in 
                      (beta_tex, 
                       layer_idx, 
                       layer_idx_proof, 
                       layer_sib_idx, 
                       layer_sib_idx_proof)))
            self.wait(1)
        self.play(FadeOut(last_even_tex), FadeOut(last_odd_tex))

class RecapSlide(Scene):
    def construct(self):
        self.next_section("Recap")
        title = Tex("Recap", font_size=56).to_edge(UP)
        self.play(Create(title))
        self.wait(1)
        last_text = title
        self.wait(3)
        texts = [
            ('- FRI achieves succinctness by halving the problem size at each step', 5),
            ('- Each step is verifiable via interaction Prover ↔ Verifier', 5),
            ('- It is transparent! No trusted setup required', 5),
        ]
        first = True
        for text, delay in texts:
            tex = Text(text, font_size=24).next_to(last_text, DOWN).to_edge(LEFT)
            if first:
                tex = tex.shift(DOWN)
            self.play(Create(tex))
            last_text = tex
            self.wait(delay)
            first = False

class ConclusionSlide(Scene):
    def construct(self):
        self.next_section("Conclusion")
        bg = Rectangle(width=15, height=3, color="#35B1B0", fill_opacity=1).to_edge(UP)
        title = Tex(r"Thank you", font_size=56).next_to(bg, DOWN).to_edge(LEFT).shift(DOWN/2)
        bar = Line(LEFT * 7, LEFT * 6, color=WHITE).next_to(title, DOWN).align_to(title, LEFT).shift(DOWN/2)
        by = Tex(r"FRI - by Thybault Alabarbe, 2024", color=GREY).scale(.9).next_to(bar, DOWN).align_to(bar, LEFT).shift(DOWN/2)
        self.add(bg)
        self.play(Create(title), Create(bar), Create(by))
        self.wait(4)

def fade_out(scene: Scene):
    animations = []
    for mobject in scene.mobjects:
        animations.append(FadeOut(mobject))
    scene.play(*animations)

class Presentation(Scene):
    def construct(self):
        scenes = [
            IntroductionSlide,
            StarksBirdsEyeViewSlide,
            InterpolationSlide,
            ConstraintsSlide,
            GeneratorSlide,
            FriIntroductionSlide,
            FriSlide,
            RecapSlide,
            ConclusionSlide
        ]
        for scene in scenes:
            scene.construct(self)
            fade_out(self)

def export(SceneClass: Type[Scene]):
    print(f"Exporting {SceneClass.__name__}")
    start_time = time()
    slide = SceneClass()
    slide.render()
    shutil.copy(slide.renderer.file_writer.movie_file_path, f"./{SceneClass.__name__}.mp4")
    open_media_file(slide.renderer.file_writer.movie_file_path) 
    print(f"Time: {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    export(Presentation)
