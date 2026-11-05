// Build stamp: embeds the git sha and dirty flag of the tree this binding was built from, plus the UTC build time.
// A build from an exported tree (no .git) sets ANCESTRA_GIT_SHA=<sha> and ANCESTRA_GIT_DIRTY=0|1; inside a git
// checkout the values come from git and the environment is ignored. With no git information the build is dirty.
use std::path::PathBuf;
use std::process::Command;

fn run(args: &[&str]) -> Option<String> {
    // None = the command failed; Some(s) = its trimmed stdout (possibly empty:
    // `git status --porcelain` prints nothing on a clean tree, and that is data).
    let out = Command::new(args[0]).args(&args[1..]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=ANCESTRA_GIT_SHA");
    println!("cargo:rerun-if-env-changed=ANCESTRA_GIT_DIRTY");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    // Rerun when a compiled input or the commit changes. Any rerun-if-changed line replaces cargo's default package
    // scan, so the inputs are listed explicitly. HEAD and the current branch ref are resolved by git itself: a linked
    // worktree's .git is a `gitdir:` file naming its own git dir (which holds its HEAD), and branch refs live in the
    // common dir. The loose ref is watched even when absent (packed): a missing path reruns every build. An exported
    // tree has no git: nothing to watch, the environment stamp wins. Dirty flips that move neither HEAD nor the ref
    // are not seen.
    for p in ["src/rust", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={p}");
    }
    let git_dir = run(&["git", "rev-parse", "--absolute-git-dir"]);
    if let (Some(git_dir), Some(common)) = (&git_dir, run(&["git", "rev-parse", "--git-common-dir"])) {
        println!("cargo:rerun-if-changed={}/HEAD", git_dir);
        let common = PathBuf::from(common);
        let common = if common.is_absolute() {
            common
        } else {
            std::env::current_dir().expect("build.rs cwd").join(common)
        };
        if let Some(r) = run(&["git", "symbolic-ref", "-q", "HEAD"]) {
            println!("cargo:rerun-if-changed={}", common.join(r).display());
        }
    }

    // A checkout of this package (git's top level is the manifest dir): sha and dirty come from git only, since an
    // exported sha cannot vouch for a checkout's tree. Anything else (an exported tree, also one unpacked inside
    // another repository): ANCESTRA_GIT_SHA / ANCESTRA_GIT_DIRTY only; no sha -> unknown, no clean flag -> dirty.
    let real = |p: String| PathBuf::from(p).canonicalize().ok();
    let top = run(&["git", "rev-parse", "--show-toplevel"]).and_then(real);
    let checkout = top.is_some() && top == std::env::var("CARGO_MANIFEST_DIR").ok().and_then(real);
    let (sha, dirty) = if checkout {
        let status = run(&["git", "status", "--porcelain"]);
        (run(&["git", "rev-parse", "HEAD"]), status.map_or(true, |s| !s.is_empty()))
    } else {
        let flag = std::env::var("ANCESTRA_GIT_DIRTY").ok();
        let sha = std::env::var("ANCESTRA_GIT_SHA").ok().filter(|s| !s.is_empty());
        (sha, !matches!(flag.as_deref(), Some("0" | "false" | "clean")))
    };
    let sha = sha.unwrap_or_else(|| "unknown".to_string());
    let built_at = match std::env::var("SOURCE_DATE_EPOCH").ok().filter(|s| !s.is_empty()) {
        Some(epoch) => {
            let arg = format!("@{epoch}");
            run(&["date", "-u", "-d", &arg, "+%Y-%m-%dT%H:%M:%SZ"])
                .expect("SOURCE_DATE_EPOCH must be a valid Unix timestamp")
        }
        None => run(&["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_else(|| "unknown".to_string()),
    };

    println!("cargo:rustc-env=ANCESTRA_BUILD_GIT_SHA={sha}");
    println!("cargo:rustc-env=ANCESTRA_BUILD_GIT_DIRTY={}", if dirty { "1" } else { "0" });
    println!("cargo:rustc-env=ANCESTRA_BUILD_BUILT_AT={built_at}");
}
