"""
Smoke test with a LEARNABLE synthetic task.

Classes across all users are drawn from a shared latent motif dictionary, so a
projection that separates one user's classes also separates an unseen user's.
That is the property the real head must have; a test made of pure noise
clusters cannot check it.

Group 99 is held out from meta-training -- mirrors leave-one-user-out.
"""
import numpy as np, torch, sys, time
torch.manual_seed(0)
sys.path.insert(0, '.')
from customization.head import CustomizationHead, EpisodicCustomizationLoss
from customization.episodes import TaskPool, EpisodeSampler, MetaAugmentor
from customization.registry import GestureRegistry

T, H, M = 120, 72, 16
rng = np.random.default_rng(0)
DICT = rng.normal(size=(M, T, H)).astype(np.float32)
NOISE_BASIS = rng.normal(size=(H, H)).astype(np.float32) * 0.06


def make_class(seed, per=10, sigma=0.25):
    r = np.random.default_rng(seed)
    i, j = r.choice(M, 2, replace=False)
    w = r.uniform(0.4, 0.9)
    core = w * DICT[i] + (1 - w) * DICT[j]
    nuis = (r.normal(size=(per, H)) @ NOISE_BASIS).astype(np.float32)   # per-SAMPLE nuisance
    return core[None] + nuis[:, None, :] + r.normal(size=(per, T, H)).astype(np.float32) * sigma


def make_group(gid, nc=10, per=10):
    e, y = [], []
    for c in range(nc):
        e.append(make_class(seed=1000 * gid + c, per=per)); y += [c + 1] * per
    return np.concatenate(e).astype(np.float32), np.array(y)


pool, corpus = TaskPool(), []
for g in range(14):                      # meta-training groups
    e, y = make_group(g)
    m = (rng.random((len(e), T)) < 0.25).astype(np.float32)
    pool.add(('user', g), e, y, m)
corpus_groups = []
for g in (80, 81, 82, 83):               # calibration split -- NOT in the pool
    e, y = make_group(g)
    grp = [e[y == c][:7] for c in np.unique(y)]
    corpus += grp
    corpus_groups.append(grp)
print("pool groups:", len(pool.groups), "| classes:", pool.n_classes_total)

null = (rng.normal(size=(400, T, H)).astype(np.float32) * 0.6) + 2.0
nullm = (rng.random((400, T)) < 0.25).astype(np.float32)
sampler = EpisodeSampler(pool, null, nullm, n_way=(3, 10), k_shot=(3, 7), seed=1)

head = CustomizationHead(backbone_dim=H, out_dim=64)
crit = EpisodicCustomizationLoss(warmup_episodes=200)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
tt = lambda x: None if x is None else torch.as_tensor(x, dtype=torch.float32)

accs = []
for step in range(1, 2001):
    ep = sampler.sample()
    zs = head(tt(ep['support_emb']), tt(ep['support_mask']))
    zq = head(tt(ep['query_emb']), tt(ep['query_mask']))
    zn = head(tt(ep['null_emb']), tt(ep['null_mask'])) if ep['null_emb'] is not None else None
    loss, parts = crit(zs, torch.as_tensor(ep['support_y']),
                       zq, torch.as_tensor(ep['query_y']),
                       ep['n_classes'], z_null=zn, step=step,
                       temperature=head.temperature)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step(); accs.append(parts['acc'])
    if step % 500 == 0:
        print(f"  step {step:4d} loss {parts['total']:.4f} q-acc {np.mean(accs[-500:]):.3f} "
              f"rep {parts['rep']:.3f} margin {parts['margin']:.3f} "
              f"null {parts['null']:.3f} radius {parts['mean_radius']:.3f}")
assert not np.isnan(parts['total']), "NaN loss"

head.eval()
e, y = make_group(99)
m = (rng.random((len(e), T)) < 0.25).astype(np.float32)
reg = GestureRegistry(head)
z, zn_e = reg.embed(e, m), reg.embed(null, nullm)
tr = {c: np.where(y == c)[0][:7] for c in np.unique(y)}
te = np.concatenate([np.where(y == c)[0][7:] for c in np.unique(y)])

reg.calibrate([reg.embed(zc, None) for zc in corpus],
              null_z=zn_e,
              corpus_groups=[[reg.embed(zc, None) for zc in g] for g in corpus_groups],
              verbose=True)
for c in np.unique(y):
    d = reg.add_gesture(f'g{c}', z[tr[c]], null_z=zn_e)
    print(f"  g{c:<3} {'ACCEPT' if d['added'] else 'REJECT'}  {d['message'][:64]}")
print("vocabulary:", reg.names, f"({len(reg.names)}/10 accepted)")
assert len(reg.names) >= 6, f"over-rejecting on a learnable task: {len(reg.names)}/10"

reg.calibrate_reject(zn_e, target_fpr=0.05)
keep = {int(n[1:]) for n in reg.names}
sel = np.isin(y[te], list(keep))
truth = [f'g{c}' for c in y[te][sel]]
acc_closed = np.mean([p == t for p, t in zip(reg.predict(z[te][sel], False)[0], truth)])
acc_open = np.mean([p == t for p, t in zip(reg.predict(z[te][sel], True)[0], truth)])
bg = np.mean([l is not None for l in reg.predict(zn_e, True)[0]])
print(f"closed-set {acc_closed:.3f} | open-set {acc_open:.3f} | bg false-accept {bg:.3f}")
assert acc_closed > 0.7, f"closed-set accuracy too low: {acc_closed}"

reg_u = GestureRegistry(head)
for c in np.unique(y):
    reg_u.add_gesture(f'g{c}', z[tr[c]], validate=False)
acc_unf = np.mean([p == t for p, t in
                   zip(reg_u.predict(z[te], False)[0], [f'g{c}' for c in y[te]])])
print(f"unfiltered (all 10) acc {acc_unf:.3f}")

dup_of = int(reg.names[0][1:])
d = reg.validate('dupe', z[tr[dup_of]], null_z=zn_e)
assert not d['accept'], "validator accepted an exact duplicate"
assert d['checks']['repeatability']['pass'], "duplicate should fail on distinguishability, not repeatability"
assert not d['checks']['distinguishability']['pass'], "duplicate not caught by distinguishability"
print("duplicate rejected via distinguishability:", d['message'][:70])

# a TIGHT cluster sitting on the background manifold: passes repeatability,
# must be caught by the null-collision check specifically
anchor = zn_e[0]
bgc = anchor[None] + np.random.default_rng(3).normal(size=(7, zn_e.shape[1])).astype(np.float32) * 0.01
bgc = bgc / np.linalg.norm(bgc, axis=1, keepdims=True)
d = reg.validate('bgtight', bgc, null_z=zn_e)
print("background-like candidate:", 'ACCEPT' if d['accept'] else 'REJECT', '|', d['message'][:60])
assert not d['accept'], "background-like candidate was accepted"
assert d['checks']['repeatability']['pass'], "should pass repeatability (it is a tight cluster)"
assert not d['checks']['null_collision']['pass'], "null-collision check did not fire"
print("   null-collision fired: density_ratio="
      f"{d['checks']['null_collision']['density_ratio']:.3f}")

before = reg.protos.copy()
t0 = time.perf_counter(); reg.add_gesture('extra', z[tr[np.unique(y)[-1]]], validate=False)
print(f"add_gesture wall-clock: {(time.perf_counter()-t0)*1000:.2f} ms")
assert np.allclose(before, reg.protos[:len(before)]), "existing prototypes moved"

reg.save('/tmp/reg'); torch.save(head.state_dict(), '/tmp/head.pt')
r2 = GestureRegistry.load('/tmp/reg', '/tmp/head.pt', backbone_dim=H)
assert r2.names == reg.names and np.allclose(r2.protos, reg.protos)

aug = MetaAugmentor(seq_len=T, rng=rng)
sr, sy = aug.synthesize(rng.normal(size=(30, T, 6)).astype(np.float32),
                        np.repeat([1, 2, 3], 10), per_class_variants=3)
assert sr.shape[1:] == (T, 6)
print(f"augmentor: {sr.shape} -> {len(np.unique(sy))} synthetic classes")
print("\nALL SMOKE TESTS PASSED")
