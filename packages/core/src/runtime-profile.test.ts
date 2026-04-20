import { afterEach, describe, expect, test } from "bun:test";
import { array } from "./array";
import { add } from "./ops";
import { resetCoreRuntimeProfile, snapshotCoreRuntimeProfile } from "./runtime-profile";

const ORIGINAL_PROFILE_FLAG = process.env.MLXTS_RUNTIME_PROFILE;

afterEach(() => {
  if (ORIGINAL_PROFILE_FLAG === undefined) {
    delete process.env.MLXTS_RUNTIME_PROFILE;
  } else {
    process.env.MLXTS_RUNTIME_PROFILE = ORIGINAL_PROFILE_FLAG;
  }
  resetCoreRuntimeProfile();
});

describe("core runtime profile", () => {
  test("captures wrapper, ffi, and free activity when enabled", () => {
    process.env.MLXTS_RUNTIME_PROFILE = "1";
    resetCoreRuntimeProfile();

    {
      using lhs = array([[1]], "float32");
      using rhs = array([[2]], "float32");
      using result = add(lhs, rhs);
      expect(result.item()).toBe(3);
    }

    const snapshot = snapshotCoreRuntimeProfile();
    expect(snapshot.enabled).toBe(true);
    expect(snapshot.outSlot.count).toBeGreaterThan(0);
    expect(snapshot.ffiInvoke.count).toBeGreaterThan(0);
    expect(snapshot.wrapperConstruct.count).toBeGreaterThan(0);
    expect(snapshot.registryRegister.count).toBeGreaterThan(0);
    expect(snapshot.explicitFree.count).toBeGreaterThan(0);
    expect(snapshot.registryUnregister.count).toBeGreaterThan(0);
    expect(snapshot.nativeFree.count).toBeGreaterThan(0);
    expect(snapshot.ffiLabels.add?.count ?? 0).toBeGreaterThan(0);
  });
});
