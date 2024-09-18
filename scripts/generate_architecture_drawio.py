#!/usr/bin/env python3
"""
Generate DisentangledTMR architecture diagram as a draw.io XML file.

Usage:
    python scripts/generate_architecture_drawio.py

Output:
    paper/fig/architecture.drawio

To use:
    1. Open paper/fig/architecture.drawio at https://app.diagrams.net/
    2. Edit visually (drag/resize/recolor)
    3. File → Export as → PDF (for paper)

All coordinates are in pixels. Modify the constants below to adjust layout.
"""

import xml.etree.ElementTree as ET
import os


# =============================================================================
# Color scheme (matching Gemini diagram style)
# =============================================================================
COLORS = {
    'action_bg':   ('#DAE8FC', '#6C8EBF'),   # Blue (fill, stroke)
    'action_box':  ('#B3D1F2', '#6C8EBF'),   # Darker blue boxes
    'identity_bg': ('#F8CECC', '#B85450'),    # Rose
    'identity_box':('#F4B8B5', '#B85450'),    # Darker rose boxes
    'decoder_bg':  ('#D5E8D4', '#82B366'),    # Green
    'decoder_box': ('#B8D8B0', '#82B366'),    # Darker green boxes
    'loss_dis':    ('#FFE6CC', '#D79B00'),    # Orange (disentanglement)
    'loss_rec':    ('#FFF2CC', '#D6B656'),    # Yellow (reconstruction)
    'gate':        ('#E6D0B8', '#A67C52'),    # Olive (fusion gates)
    'output_box':  ('#E1D5E7', '#9673A6'),    # Purple (outputs/H vectors)
    'input_box':   ('#F5F5F5', '#666666'),    # Gray (inputs)
    'mixformer':   ('#C8DCC8', '#6C8E6C'),    # Muted green (MixFormer)
    'white':       ('#FFFFFF', '#000000'),
}


# =============================================================================
# Draw.io XML Builder
# =============================================================================
class DrawioBuilder:
    """Builds draw.io mxGraph XML programmatically."""

    def __init__(self, page_w=1950, page_h=1150):
        self.page_w = page_w
        self.page_h = page_h
        self._cells = []
        self._id = 2  # 0, 1 reserved by draw.io

    def _nid(self):
        cid = str(self._id)
        self._id += 1
        return cid

    # --- Vertex elements ---

    def region(self, x, y, w, h, label='', color_key='white',
               font_size=13, dashed=True, opacity=40):
        """Large dashed-border background region."""
        fill, stroke = COLORS[color_key]
        cid = self._nid()
        style = (
            f'rounded=1;whiteSpace=wrap;html=1;'
            f'fillColor={fill};strokeColor={stroke};'
            f'fontSize={font_size};fontStyle=1;'
            f'verticalAlign=top;align=left;'
            f'spacingLeft=10;spacingTop=5;'
            f'opacity={opacity};'
            + ('dashed=1;dashPattern=8 4;' if dashed else '')
        )
        self._cells.append(('v', cid, label, style, x, y, w, h))
        return cid

    def box(self, x, y, w, h, label, color_key='white',
            font_size=10, bold=False, rounded=1, align='center'):
        """Standard box."""
        fill, stroke = COLORS[color_key]
        fs = 1 if bold else 0
        cid = self._nid()
        style = (
            f'rounded={rounded};whiteSpace=wrap;html=1;'
            f'fillColor={fill};strokeColor={stroke};'
            f'fontSize={font_size};fontStyle={fs};'
            f'verticalAlign=middle;align={align};'
        )
        self._cells.append(('v', cid, label, style, x, y, w, h))
        return cid

    def label(self, x, y, w, h, text, font_size=9, bold=False, color='#333333',
              align='center', valign='middle'):
        """Free-floating text label (no border/fill)."""
        fs = 1 if bold else 0
        cid = self._nid()
        style = (
            f'text;html=1;strokeColor=none;fillColor=none;'
            f'fontSize={font_size};fontStyle={fs};fontColor={color};'
            f'align={align};verticalAlign={valign};'
            f'whiteSpace=wrap;'
        )
        self._cells.append(('v', cid, text, style, x, y, w, h))
        return cid

    # --- Edge elements ---

    def arrow(self, src, tgt, label='', color='#333333', width=1.5,
              dashed=False, curved=False, font_size=8):
        """Arrow from src to tgt."""
        cid = self._nid()
        edge_style = 'edgeStyle=elbowEdgeStyle;elbow=vertical;' if curved else 'edgeStyle=orthogonalEdgeStyle;'
        style = (
            f'{edge_style}'
            f'rounded=0;orthogonalLoop=1;jettySize=auto;html=1;'
            f'strokeColor={color};strokeWidth={width};'
            f'fontSize={font_size};'
            + ('dashed=1;' if dashed else '')
        )
        self._cells.append(('e', cid, label, style, src, tgt))
        return cid

    def arrow_styled(self, src, tgt, label='', style_str=''):
        """Arrow with fully custom style string."""
        cid = self._nid()
        self._cells.append(('e', cid, label, style_str, src, tgt))
        return cid

    # --- XML output ---

    def to_xml(self):
        mxfile = ET.Element('mxfile', host='app.diagrams.net')
        diag = ET.SubElement(mxfile, 'diagram', id='arch',
                             name='DisentangledTMR Architecture')
        model = ET.SubElement(diag, 'mxGraphModel',
                              dx='1422', dy='762', grid='1', gridSize='10',
                              guides='1', tooltips='1', connect='1', arrows='1',
                              fold='1', page='1', pageScale='1',
                              pageWidth=str(self.page_w),
                              pageHeight=str(self.page_h))
        root = ET.SubElement(model, 'root')
        ET.SubElement(root, 'mxCell', id='0')
        ET.SubElement(root, 'mxCell', id='1', parent='0')

        for cell in self._cells:
            kind = cell[0]
            if kind == 'v':
                _, cid, val, style, x, y, w, h = cell
                mc = ET.SubElement(root, 'mxCell',
                                   id=cid, value=val, style=style,
                                   vertex='1', parent='1')
                geom = ET.SubElement(mc, 'mxGeometry',
                                     x=str(x), y=str(y),
                                     width=str(w), height=str(h))
                geom.set('as', 'geometry')
            elif kind == 'e':
                _, cid, val, style, src, tgt = cell
                mc = ET.SubElement(root, 'mxCell',
                                   id=cid, value=val, style=style,
                                   edge='1', source=src, target=tgt,
                                   parent='1')
                geom = ET.SubElement(mc, 'mxGeometry', relative='1')
                geom.set('as', 'geometry')

        # Pretty-print
        ET.indent(mxfile, space='  ')
        return ET.tostring(mxfile, encoding='unicode', xml_declaration=True)

    def save(self, path):
        xml = self.to_xml()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(xml)
        print(f'Saved: {path}')


# =============================================================================
# Build the architecture diagram
# =============================================================================
def build_diagram():
    d = DrawioBuilder(page_w=1950, page_h=1150)

    # =========================================================================
    # TITLE
    # =========================================================================
    d.label(600, 0, 700, 30,
            '<b>DisentangledTMR Architecture</b> — Total model: 22.7M parameters',
            font_size=14, bold=True)

    # =========================================================================
    # BACKGROUND REGIONS
    # =========================================================================
    ae_bg = d.region(200, 40, 730, 420,
                     '<b>Action Encoder E<sub>A</sub></b> (4.9M params, output: T × 768)',
                     color_key='action_bg', font_size=12, opacity=30)

    ie_bg = d.region(200, 490, 730, 280,
                     '<b>Identity Encoder E<sub>I</sub></b> (0.8M params, output: 256)',
                     color_key='identity_bg', font_size=12, opacity=30)

    dec_bg = d.region(1100, 40, 520, 770,
                      '<b>Factorized Decoder D</b> (17.0M params, 6 layers, d<sub>model</sub>=320)',
                      color_key='decoder_bg', font_size=12, opacity=30)

    # =========================================================================
    # INPUTS (left side)
    # =========================================================================
    d.label(10, 40, 170, 20, '<b>INPUTS</b>', font_size=12, bold=True, align='left')

    source = d.box(30, 80, 140, 70,
                   '<b>Source:</b> Person A<br>Action X<br><i>(B, 3, 64, 25, 1)</i>',
                   color_key='input_box', font_size=9)

    target = d.box(30, 530, 140, 70,
                   '<b>Target:</b> Person B<br>(any action)<br><i>(B, 3, 64, 25, 1)</i>',
                   color_key='input_box', font_size=9)

    # =========================================================================
    # ACTION ENCODER internals
    # =========================================================================
    # Row 1: Input Processing → Multi-Scale Conv
    input_proc = d.box(220, 100, 120, 55,
                       '<b>Input Processing</b><br>pos + vel + accel<br>9 channels',
                       color_key='action_box', font_size=9)

    # Input projection (this happens FIRST in code)
    input_proj = d.box(220, 175, 120, 40,
                       '<b>Input Projection</b><br>Linear: 225 → 256',
                       color_key='action_box', font_size=8)

    # Multi-scale temporal convolutions (AFTER projection)
    conv_k3 = d.box(220, 235, 80, 30, 'Conv1D<br>k=3',
                    color_key='action_box', font_size=8)
    conv_k5 = d.box(220, 275, 80, 30, 'Conv1D<br>k=5',
                    color_key='action_box', font_size=8)
    conv_k7 = d.box(220, 315, 80, 30, 'Conv1D<br>k=7',
                    color_key='action_box', font_size=8)
    d.label(210, 350, 100, 20, 'Multi-Scale<br>Temporal Conv',
            font_size=7, color='#555555')

    # Fusion conv (combines the 3 conv outputs)
    fusion_conv = d.box(330, 260, 80, 40, 'Fusion<br>Conv1D',
                        color_key='action_box', font_size=8)

    # Temporal Attention
    temp_attn = d.box(440, 200, 130, 55,
                      '<b>Temporal Attention</b><br>2 layers, 8 heads<br>256-dim',
                      color_key='action_box', font_size=9)

    # Linear projection (no LSTM — stable model has no_lstm=True)
    linear_proj = d.box(600, 200, 110, 45,
                        'Linear Projection<br>256 → 768',
                        color_key='action_box', font_size=9)

    # MixFormer backbone (parallel path)
    mixformer = d.box(310, 385, 165, 50,
                      '<b>Skeleton MixFormer</b><br><b>Backbone</b><br>(2.3M params, 320-dim)',
                      color_key='mixformer', font_size=9)

    mix_linear = d.box(510, 390, 100, 40,
                       'Linear<br>320 → 768',
                       color_key='mixformer', font_size=9)

    # Gated Fusion (merges LSTM path and MixFormer path)
    gated_fusion = d.box(730, 260, 180, 65,
                         '<b>Gated Fusion</b><br>'
                         'g · H<sub>attn</sub> + (1−g) · H<sub>MixF</sub><br>'
                         'g = σ(W[H<sub>attn</sub>, H<sub>MixF</sub>])',
                         color_key='gate', font_size=9)

    # Asymmetric capacity annotation
    d.label(440, 445, 300, 20,
            '<i>Asymmetric capacity: 768 vs. 256 (information bottleneck)</i>',
            font_size=8, color='#666666')

    # H_action output
    h_action = d.box(960, 195, 120, 50,
                     '<b>H<sub>action</sub></b><br>∈ ℝ<sup>T × 768</sup>',
                     color_key='output_box', font_size=10, bold=True)

    # --- Action Encoder arrows ---
    d.arrow(source, input_proc)
    d.arrow(input_proc, input_proj)
    d.arrow(input_proj, conv_k3)
    d.arrow(input_proj, conv_k5)
    d.arrow(input_proj, conv_k7)
    d.arrow(conv_k3, fusion_conv)
    d.arrow(conv_k5, fusion_conv)
    d.arrow(conv_k7, fusion_conv)
    d.arrow(fusion_conv, temp_attn)
    d.arrow(temp_attn, linear_proj)
    d.arrow(linear_proj, gated_fusion)
    d.arrow(source, mixformer, dashed=True, color='#6C8E6C')
    d.arrow(mixformer, mix_linear)
    d.arrow(mix_linear, gated_fusion)
    d.arrow(gated_fusion, h_action)

    # =========================================================================
    # IDENTITY ENCODER internals
    # =========================================================================
    # Static pose extraction
    static_pose = d.box(220, 530, 120, 55,
                        '<b>Static Pose</b><br><b>Extraction</b><br>'
                        'mean over T frames',
                        color_key='identity_box', font_size=9)

    # Spatial GCN
    spatial_gcn = d.box(370, 530, 110, 50,
                        '<b>Spatial GCN</b><br>3 layers<br>64→128→256→256',
                        color_key='identity_box', font_size=8)

    # Spatial Self-Attention
    spatial_attn = d.box(510, 530, 110, 50,
                         '<b>Spatial</b><br><b>Self-Attention</b><br>8 heads, 256-dim',
                         color_key='identity_box', font_size=8)

    # Global Avg Pool
    global_pool = d.box(650, 540, 80, 35,
                        'Global<br>Avg Pool',
                        color_key='identity_box', font_size=8)

    # Bone Length Encoder (parallel path at bottom)
    bone_enc = d.box(370, 640, 130, 45,
                     '<b>Bone Length Encoder</b><br>MLP: 24→128→128',
                     color_key='identity_box', font_size=8)

    # Concat
    concat = d.box(650, 610, 80, 35,
                   '<b>Concat</b><br>[256; 128]',
                   color_key='identity_box', font_size=8)

    # Fusion MLP
    fusion_mlp = d.box(760, 560, 110, 50,
                       '<b>Fusion MLP</b><br>384 → 512 → 256',
                       color_key='identity_box', font_size=9)

    # H_identity output
    h_identity = d.box(960, 565, 120, 50,
                       '<b>H<sub>identity</sub></b><br>∈ ℝ<sup>256</sup>',
                       color_key='output_box', font_size=10, bold=True)

    # --- Identity Encoder arrows ---
    d.arrow(target, static_pose)
    d.arrow(static_pose, spatial_gcn)
    d.arrow(spatial_gcn, spatial_attn)
    d.arrow(spatial_attn, global_pool)
    d.arrow(global_pool, concat)
    d.arrow(static_pose, bone_enc, dashed=True, color='#B85450')
    d.arrow(bone_enc, concat)
    d.arrow(concat, fusion_mlp)
    d.arrow(fusion_mlp, h_identity)

    # =========================================================================
    # FACTORIZED DECODER internals
    # =========================================================================
    # Input: previous frame embedding
    dec_input = d.box(1270, 75, 140, 40,
                      'Frame Embedding<br>Linear: 75 → 320',
                      color_key='decoder_box', font_size=9)

    # Causal Self-Attention
    causal_sa = d.box(1270, 150, 140, 45,
                      '<b>Causal Self-Attention</b><br>8 heads',
                      color_key='decoder_box', font_size=9)

    # Cross-Attention (Action) - left
    xattn_action = d.box(1140, 250, 155, 55,
                         '<b>Cross-Attention</b><br><b>(Action)</b><br>'
                         'Q: Z<sub>dec</sub>,  K,V: H<sub>action</sub>',
                         color_key='decoder_box', font_size=9)

    # Cross-Attention (Identity) - right
    xattn_identity = d.box(1390, 250, 155, 55,
                           '<b>Cross-Attention</b><br><b>(Identity)</b><br>'
                           'Q: Z<sub>dec</sub>,  K,V: H<sub>identity</sub>',
                           color_key='decoder_box', font_size=9)

    # Adaptive Fusion Gate
    adaptive_gate = d.box(1200, 365, 280, 65,
                          '<b>Adaptive Fusion Gate α</b><br>'
                          'Z = α · Z<sub>A</sub> + (1−α) · Z<sub>I</sub><br>'
                          'α = sigmoid(W[Z<sub>A</sub>, Z<sub>I</sub>])',
                          color_key='gate', font_size=10, bold=True)

    # LayerNorm
    layer_norm = d.box(1290, 450, 100, 30,
                       'LayerNorm',
                       color_key='decoder_box', font_size=9)

    # FFN
    ffn = d.box(1250, 510, 180, 45,
                '<b>FFN</b><br>320 → 2048 → 320',
                color_key='decoder_box', font_size=10)

    # Repeat annotation
    d.label(1540, 510, 60, 40, '×6<br>layers',
            font_size=11, bold=True, color='#82B366')

    # Output Projection
    output_proj = d.box(1250, 600, 180, 45,
                        '<b>Output Projection</b><br>Linear: 320 → 75 (3×25×1)',
                        color_key='decoder_box', font_size=9)

    # --- Decoder arrows ---
    d.arrow(dec_input, causal_sa)
    d.arrow(causal_sa, xattn_action)
    d.arrow(causal_sa, xattn_identity)
    d.arrow(xattn_action, adaptive_gate)
    d.arrow(xattn_identity, adaptive_gate)
    d.arrow(adaptive_gate, layer_norm)
    d.arrow(layer_norm, ffn)
    d.arrow(ffn, output_proj)

    # --- H_action → Decoder arrow (prominent) ---
    d.arrow(h_action, xattn_action, color='#2980B9', width=2.5,
            label='')

    # --- H_identity → Decoder arrow (prominent, this is the key flow!) ---
    d.arrow(h_identity, xattn_identity, color='#B85450', width=2.5,
            label='')

    # Annotation: H_identity is broadcast
    d.label(1050, 310, 80, 40,
            '<i>broadcast<br>(B,256) →<br>(T,B,320)</i>',
            font_size=7, color='#B85450')

    # Annotation: stop-gradient
    d.label(920, 280, 100, 25,
            '<i>stop-gradient<br>to discriminator</i>',
            font_size=7, color='#2980B9')

    # =========================================================================
    # OUTPUT (right side) — single output: target identity + source action
    # =========================================================================
    output = d.box(1680, 580, 160, 75,
                   '<b>Output:</b><br>Person B\'s body,<br>Action X<br>'
                   '<i>(B, 3, 63, 25, 1)</i>',
                   color_key='input_box', font_size=9)

    d.label(1680, 510, 160, 30,
            '<i>Retargeted motion:<br>source action + target identity</i>',
            font_size=7, color='#555555')

    d.arrow(output_proj, output)

    # =========================================================================
    # DISENTANGLEMENT LOSSES (bottom center)
    # =========================================================================
    d.label(350, 795, 250, 20,
            '<b>Disentanglement Losses (Stage 1 &amp; 3)</b>',
            font_size=10, bold=True, color='#D79B00')

    l_ctr = d.box(220, 825, 90, 40,
                  'L<sub>ctr</sub><br><i>(Contrastive/<br>SupCon)</i>',
                  color_key='loss_dis', font_size=7)
    l_adv = d.box(325, 825, 90, 40,
                  'L<sub>adv</sub><br><i>(Adversarial)</i>',
                  color_key='loss_dis', font_size=7)
    l_orth = d.box(430, 825, 90, 40,
                   'L<sub>orth</sub><br><i>(Orthogonality)</i>',
                   color_key='loss_dis', font_size=7)
    l_mi = d.box(535, 825, 90, 40,
                 'L<sub>MI</sub><br><i>(Mutual Info)</i>',
                 color_key='loss_dis', font_size=7)

    # AR and RI classifiers
    ar_cls = d.box(660, 825, 100, 40,
                   '<b>AR Classifier</b><br><i>(3-layer MLP)</i>',
                   color_key='loss_dis', font_size=7)
    ri_cls = d.box(775, 825, 100, 40,
                   '<b>RI Classifier</b><br><i>(3-layer MLP)</i>',
                   color_key='loss_dis', font_size=7)

    # L_AR and L_RI
    l_ar = d.box(660, 875, 100, 25, 'L<sub>AR</sub> (cross-entropy)',
                 color_key='loss_dis', font_size=7)
    l_ri = d.box(775, 875, 100, 25, 'L<sub>RI</sub> (cross-entropy)',
                 color_key='loss_dis', font_size=7)

    # =========================================================================
    # RECONSTRUCTION LOSSES (bottom right)
    # =========================================================================
    d.label(980, 795, 300, 20,
            '<b>Reconstruction &amp; Physical Losses (Stage 2 &amp; 3)</b>',
            font_size=10, bold=True, color='#D6B656')

    rec_labels = [
        ('L<sub>MSE</sub><br><i>(position)</i>', 940),
        ('L<sub>bone</sub><br><i>(bone length)</i>', 1030),
        ('L<sub>smooth</sub><br><i>(acceleration)</i>', 1120),
        ('L<sub>vel</sub><br><i>(velocity)</i>', 1210),
        ('L<sub>EE</sub><br><i>(end-effector)</i>', 1300),
        ('L<sub>joint</sub><br><i>(joint limits)</i>', 1390),
        ('L<sub>foot</sub><br><i>(foot contact)</i>', 1480),
    ]
    for lbl, lx in rec_labels:
        d.box(lx, 825, 80, 40, lbl, color_key='loss_rec', font_size=7)

    # =========================================================================
    # CONNECTION ANNOTATIONS
    # =========================================================================
    # Arrow from H_action area down to loss classifiers (dashed, indicating
    # the disentanglement losses operate on encoder outputs)
    d.arrow(h_action, ar_cls, dashed=True, color='#D79B00', width=1)
    d.arrow(h_identity, ri_cls, dashed=True, color='#D79B00', width=1)

    return d


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    diagram = build_diagram()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'paper', 'fig', 'architecture.drawio')
    out_path = os.path.abspath(out_path)
    diagram.save(out_path)
    print(f'\nTo edit: open {out_path} at https://app.diagrams.net/')
    print('To export: File → Export as → PDF')
