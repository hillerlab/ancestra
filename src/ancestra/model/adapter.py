"""23-state joint posteriors + site bits -> CDS decoder emissions ``(la, lb, c)`` per strand plane, and the native
UTR/TSS/TES channels.

The decoder (``ancestra._native.decode(seq, strand, la, lb, c, start, params)``) takes, per strand and in genome
coordinates:
  - ``la [3, W]``: log p(non-coding), log p(CDS), log p(CDS intron);
  - ``lb [3, W]``: log p(codon position | CDS or CDS intron): the codon position of a CDS base (0-based in transcript
    order) and, on CDS-intron bases, the upstream CDS length mod 3 (the number of codon bases pending at the donor).
    The adapter returns that conditional: ``lb[r] = log Σ_{s∈Φ_r} p(s) - log Σ_{s∈CDS∪intron} p(s)``;
  - ``c [4, W]``: probabilities of donor, acceptor, START, STOP, straight from the site bits 0-3 (sigmoid of the
    logits). TSS (bit 4) and TES (bit 5) are not used by the decoder.

The 23 states; "pos" is the base's codon position in transcript order, "base" says which base the state marks, in
transcript order:

| id | state | base it marks | la class | lb phase |
|---|---|---|---|---|
| 0 | intergenic | - | non-coding | - |
| 1 | CDS intron, 2 bases pending (after EI 8) | intron base | intron | 2 |
| 2 | CDS intron, 0 pending (after EI 9) | intron base | intron | 0 |
| 3 | CDS intron, 1 pending (after EI 10) | intron base | intron | 1 |
| 4 / 5 / 6 | CDS exon, pos 0 / 1 / 2 | exon base | CDS | 0 / 1 / 2 |
| 7 | START | first base (A) of the ATG; pos 0 | CDS | 0 |
| 8 / 9 / 10 | exon -> intron (EI) | LAST exon base before the intron; pos 1 / 2 / 0 | CDS | 1 / 2 / 0 |
| 11 / 12 / 13 | intron -> exon (IE) | FIRST exon base after the intron; pos 2 / 0 / 1 | CDS | 2 / 0 / 1 |
| 14 | STOP | LAST (third) base of the stop codon; pos 2 | CDS | 2 |
| 15 / 17 | 5' / 3' UTR exon | exon base | non-coding | - |
| 16 / 18 | 5' / 3' UTR intron | intron base | non-coding | - |
| 19 / 21 | 5' / 3' UTR exon -> intron | last UTR exon base before the intron | non-coding | - |
| 20 / 22 | 5' / 3' UTR intron -> exon | first UTR exon base after the intron | non-coding | - |

The stop codon is CDS (the decoder's CDS chains include it and close a gene at ``stop + 3``). The site bits mark:
donor = the first intron base, acceptor = the last intron base, START = the A of the ATG, STOP = the first base of the
stop codon (all in transcript order).
"""
import torch

LOG_FLOOR = -69.0          # a finite stand-in for log(0), as in ancestra.decode.viterbi

NONCODING = (0, 15, 16, 17, 18, 19, 20, 21, 22)
CDS = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
INTRON = (1, 2, 3)
# lb phase of each CDS / CDS-intron state (table above)
PHASE = {1: 2, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2, 7: 0, 8: 1, 9: 2, 10: 0, 11: 2, 12: 0, 13: 1, 14: 2}
PHASE_SETS = tuple(tuple(s for s in sorted(PHASE) if PHASE[s] == r) for r in range(3))
CODING = tuple(sorted(PHASE))


def _lse(x, idx, dim):
    return torch.logsumexp(x.index_select(dim, torch.tensor(idx, device=x.device)), dim=dim)


@torch.no_grad()
def adapt(joint_logp, site_p):
    """Joint log-posteriors ``[..., 23, W]`` + site probabilities ``[..., 6, W]`` -> f32 ``(la [...,3,W], lb [...,3,W], c [...,4,W])``.

    Works on any leading shape (e.g. ``[2, 23, W]`` for both strand planes). ``joint_logp`` must be normalised over the 23 states
    (log_softmax, or ``_log_mean2`` of two views). Non-finite values are replaced by LOG_FLOOR / 0.
    """
    x = joint_logp.float()
    x = torch.nan_to_num(x, nan=LOG_FLOOR, neginf=LOG_FLOOR, posinf=0.0).clamp_min(LOG_FLOOR)
    d = x.dim() - 2
    la = torch.stack([_lse(x, NONCODING, d), _lse(x, CDS, d), _lse(x, INTRON, d)], dim=d)
    coding = _lse(x, CODING, d)
    lb = torch.stack([_lse(x, PHASE_SETS[r], d) - coding for r in range(3)], dim=d)
    c = torch.nan_to_num(site_p.float().narrow(d, 0, 4), nan=0.0).clamp(0.0, 1.0)
    return la.float(), lb.float(), c.float()


#: State groups of the utr7 state channels 0-4: intergenic; 5'UTR exon; 5'UTR intron; 3'UTR exon; 3'UTR intron.
UTR7_GROUPS = ((0,), (15, 19, 20), (16,), (17, 21, 22), (18,))
UTR7_TSS, UTR7_TES = 4, 5  # site bits of the utr7 end channels 5-6


@torch.no_grad()
def native_utr7(joint_logp, site_p):
    """Native UTR/TSS/TES channels ``[..., 7, W]`` f32 from the same inputs as ``adapt``.

    Channels 0-4 are log p over the state groups ``UTR7_GROUPS`` (logsumexp of the normalised joint log-posteriors, so
    channels 0-4 plus the remaining states' mass logsumexp to 0). Channels 5-6 are the TSS/TES logits (site bits 4-5),
    i.e. ``logit`` of the (TTA-averaged) site probabilities, so ``sigmoid`` inverts them back. A per-position,
    per-plane transform, so the strand-plane orientation matches ``adapt``'s outputs exactly.
    """
    x = joint_logp.float()
    x = torch.nan_to_num(x, nan=LOG_FLOOR, neginf=LOG_FLOOR, posinf=0.0).clamp_min(LOG_FLOOR)
    d = x.dim() - 2
    states = torch.stack([_lse(x, g, d) for g in UTR7_GROUPS], dim=d)
    p = torch.nan_to_num(site_p.float(), nan=0.0).clamp(0.0, 1.0)
    ends = torch.logit(p.index_select(d, torch.tensor([UTR7_TSS, UTR7_TES], device=p.device)), eps=1e-6)
    return torch.cat([states, ends], dim=d).float()
