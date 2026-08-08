(function () {
  "use strict";

  const VIEWER_SCHEMA = "contraption.viewer/v2";
  const SCENE_SCHEMA = "contraption.physical-scene/v1";
  const HASH_PATTERN = /^sha256:[0-9a-f]{64}$/;
  const byId = (id) => document.getElementById(id);

  function fail(message) {
    throw new Error(`Canonical assembly scene rejected: ${message}`);
  }

  function requireCondition(condition, message) {
    if (!condition) fail(message);
  }

  function requireObject(value, label) {
    requireCondition(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
    return value;
  }

  function requireKeys(value, required, optional, label) {
    required.forEach((key) => requireCondition(Object.prototype.hasOwnProperty.call(value, key),
      `${label} is missing required field ${key}`));
    const allowed = new Set(required.concat(optional));
    const unknown = Object.keys(value).filter((key) => !allowed.has(key));
    requireCondition(unknown.length === 0,
      `${label} contains fields the viewer would ignore: ${unknown.join(", ")}`);
  }

  function requireArray(value, label, nonempty) {
    requireCondition(Array.isArray(value), `${label} must be an array`);
    if (nonempty) requireCondition(value.length > 0, `${label} must not be empty`);
    return value;
  }

  function requireString(value, label) {
    requireCondition(typeof value === "string" && value.trim().length > 0, `${label} must be a non-empty string`);
    return value;
  }

  function requireNumber(value, label, positive) {
    requireCondition(typeof value === "number" && Number.isFinite(value), `${label} must be finite`);
    if (positive) requireCondition(value > 0, `${label} must be greater than zero`);
    return value;
  }

  function requireVector(value, label, length) {
    const result = requireArray(value, label, false);
    requireCondition(result.length === length, `${label} must contain exactly ${length} numbers`);
    result.forEach((item, index) => requireNumber(item, `${label}[${index}]`, false));
    return result;
  }

  function requirePose(value, label) {
    const pose = requireObject(value, label);
    requireKeys(pose, ["translation_m", "rotation_quaternion_wxyz"], [], label);
    requireVector(pose.translation_m, `${label}.translation_m`, 3);
    const quaternion = requireVector(pose.rotation_quaternion_wxyz, `${label}.rotation_quaternion_wxyz`, 4);
    const norm = Math.sqrt(quaternion.reduce((sum, item) => sum + item * item, 0));
    requireCondition(Math.abs(norm - 1) <= 1e-9, `${label}.rotation_quaternion_wxyz must be normalized`);
    const firstNonzero = quaternion.find((item) => Math.abs(item) > 1e-15) || 0;
    requireCondition(firstNonzero >= 0,
      `${label}.rotation_quaternion_wxyz must use the canonical quaternion sign`);
    return pose;
  }

  function validateGeometry(geometry, label) {
    requireObject(geometry, label);
    const kind = requireString(geometry.kind, `${label}.kind`);
    if (kind === "box" || kind === "sphere" || kind === "cylinder") {
      requireKeys(geometry, ["kind", "dimensions_m", "mesh_uri"], [], label);
      requireVector(geometry.dimensions_m, `${label}.dimensions_m`, 3).forEach((item, index) => {
        requireCondition(item > 0, `${label}.dimensions_m[${index}] must be greater than zero`);
      });
      requireCondition(geometry.mesh_uri === null, `${label}.mesh_uri must be null for ${kind}`);
    } else if (kind === "mesh") {
      fail(`${label} uses mesh geometry whose exact data is not embedded; refusing a bounding-box substitute`);
    } else {
      fail(`${label}.kind ${JSON.stringify(kind)} is unsupported`);
    }
  }

  function validatedPayload(candidate) {
    let payload = candidate;
    if (candidate === undefined) {
      const payloadNode = byId("contraption-data");
      requireCondition(payloadNode !== null, "embedded payload element is missing");
      try {
        payload = JSON.parse(payloadNode.textContent);
      } catch (error) {
        fail(`embedded payload is not valid JSON (${error.message})`);
      }
    }
    requireObject(payload, "payload");
    requireKeys(payload, ["schema", "title", "assembly_sha256", "scene"], ["live"], "payload");
    requireString(payload.title, "payload.title");
    requireCondition(payload.schema === VIEWER_SCHEMA, `payload.schema must be ${VIEWER_SCHEMA}`);
    requireCondition(HASH_PATTERN.test(payload.assembly_sha256), "payload.assembly_sha256 is not canonical");
    const scene = requireObject(payload.scene, "payload.scene");
    requireKeys(scene,
      ["schema", "assembly_sha256", "contraption_id", "components", "connections", "body_poses", "connector_poses"],
      ["body_pose_frames"], "scene");
    requireCondition(scene.schema === SCENE_SCHEMA, `scene.schema must be ${SCENE_SCHEMA}`);
    requireCondition(HASH_PATTERN.test(scene.assembly_sha256), "scene.assembly_sha256 is not canonical");
    requireCondition(payload.assembly_sha256 === scene.assembly_sha256,
      `assembly hash mismatch (${payload.assembly_sha256} != ${scene.assembly_sha256})`);
    requireString(scene.contraption_id, "scene.contraption_id");
    if (payload.live !== undefined) {
      const live = requireObject(payload.live, "payload.live");
      requireKeys(live, ["schema_endpoint", "simulate_endpoint"], [], "payload.live");
      ["schema_endpoint", "simulate_endpoint"].forEach((name) => {
        const endpoint = requireString(live[name], `payload.live.${name}`);
        requireCondition(endpoint.startsWith("/") && !endpoint.startsWith("//") &&
          !endpoint.includes("\\") && !endpoint.includes("?") && !endpoint.includes("#"),
        `payload.live.${name} must be a same-origin absolute path`);
      });
    }

    const components = requireArray(scene.components, "scene.components", true);
    const instanceIds = new Set();
    const bodyKeys = new Set();
    const connectorKeys = new Set();
    const spatialConnectorKeys = new Set();
    let solidCount = 0;
    components.forEach((component, componentIndex) => {
      const componentLabel = `scene.components[${componentIndex}]`;
      requireObject(component, componentLabel);
      requireKeys(component, ["id", "package", "model", "physical_role", "bodies", "connectors"], [], componentLabel);
      const instanceId = requireString(component.id, `${componentLabel}.id`);
      requireCondition(!instanceId.includes("/"), `${componentLabel}.id cannot contain '/'`);
      requireCondition(!instanceIds.has(instanceId), `duplicate component id ${JSON.stringify(instanceId)}`);
      instanceIds.add(instanceId);
      requireString(component.package, `${componentLabel}.package`);
      requireString(component.model, `${componentLabel}.model`);
      requireCondition(["part", "boundary", "software"].includes(component.physical_role),
        `${componentLabel}.physical_role is unsupported`);
      const bodies = requireArray(component.bodies, `${componentLabel}.bodies`, component.physical_role === "part");
      requireCondition(component.physical_role === "part" || bodies.length === 0,
        `${componentLabel} is nonspatial but declares bodies`);
      const bodyIds = new Set();
      bodies.forEach((body, bodyIndex) => {
        const bodyLabel = `${componentLabel}.bodies[${bodyIndex}]`;
        requireObject(body, bodyLabel);
        requireKeys(body, ["id", "local_pose", "solids"], [], bodyLabel);
        const bodyId = requireString(body.id, `${bodyLabel}.id`);
        requireCondition(!bodyId.includes("/"), `${bodyLabel}.id cannot contain '/'`);
        requireCondition(!bodyIds.has(bodyId), `${componentLabel} has duplicate body_id ${JSON.stringify(bodyId)}`);
        bodyIds.add(bodyId);
        bodyKeys.add(`${instanceId}/${bodyId}`);
        requirePose(body.local_pose, `${bodyLabel}.local_pose`);
        const solidIds = new Set();
        requireArray(body.solids, `${bodyLabel}.solids`, true).forEach((solid, solidIndex) => {
          const solidLabel = `${bodyLabel}.solids[${solidIndex}]`;
          requireObject(solid, solidLabel);
          requireKeys(solid, ["id", "geometry", "local_pose", "provenance"], [], solidLabel);
          const solidId = requireString(solid.id, `${solidLabel}.id`);
          requireCondition(!solidIds.has(solidId), `${bodyLabel} has duplicate solid_id ${JSON.stringify(solidId)}`);
          solidIds.add(solidId);
          validateGeometry(solid.geometry, `${solidLabel}.geometry`);
          requirePose(solid.local_pose, `${solidLabel}.local_pose`);
          const provenance = requireObject(solid.provenance, `${solidLabel}.provenance`);
          requireKeys(provenance, ["kind", "source", "reference"], [], `${solidLabel}.provenance`);
          requireString(provenance.kind, `${solidLabel}.provenance.kind`);
          requireString(provenance.source, `${solidLabel}.provenance.source`);
          if (provenance.reference !== null) {
            requireString(provenance.reference, `${solidLabel}.provenance.reference`);
          }
          solidCount += 1;
        });
      });
      const connectorIds = new Set();
      requireArray(component.connectors, `${componentLabel}.connectors`, false).forEach((connector, connectorIndex) => {
        const connectorLabel = `${componentLabel}.connectors[${connectorIndex}]`;
        requireObject(connector, connectorLabel);
        requireKeys(connector,
          ["id", "model_port", "body", "domain", "interface", "local_pose", "provenance", "joint_coordinate_state"],
          [], connectorLabel);
        const connectorId = requireString(connector.id, `${connectorLabel}.id`);
        requireCondition(!connectorIds.has(connectorId),
          `${componentLabel} has duplicate connector id ${JSON.stringify(connectorId)}`);
        connectorIds.add(connectorId);
        connectorKeys.add(`${instanceId}.${connectorId}`);
        if (connector.model_port !== null) requireString(connector.model_port, `${connectorLabel}.model_port`);
        if (connector.joint_coordinate_state !== null) {
          requireString(connector.joint_coordinate_state, `${connectorLabel}.joint_coordinate_state`);
          requireCondition(connector.interface === "rotational-shaft",
            `${connectorLabel}.joint_coordinate_state is only valid for rotational-shaft`);
        }
        requireString(connector.domain, `${connectorLabel}.domain`);
        requireString(connector.interface, `${connectorLabel}.interface`);
        if (connector.body === null) {
          requireCondition(connector.local_pose === null,
            `${connectorLabel} with null body must have null local_pose`);
          requireCondition(component.physical_role !== "part",
            `${connectorLabel} on a part must be spatial`);
        } else {
          requireString(connector.body, `${connectorLabel}.body`);
          requireCondition(bodyIds.has(connector.body),
            `${connectorLabel}.body references unknown body ${JSON.stringify(connector.body)}`);
          requirePose(connector.local_pose, `${connectorLabel}.local_pose`);
          requireCondition(component.physical_role === "part",
            `${connectorLabel} on a nonspatial component cannot be spatial`);
          spatialConnectorKeys.add(`${instanceId}.${connectorId}`);
        }
        const provenance = requireObject(connector.provenance, `${connectorLabel}.provenance`);
        requireKeys(provenance, ["kind", "source", "reference"], [], `${connectorLabel}.provenance`);
        requireString(provenance.kind, `${connectorLabel}.provenance.kind`);
        requireString(provenance.source, `${connectorLabel}.provenance.source`);
        if (provenance.reference !== null) {
          requireString(provenance.reference, `${connectorLabel}.provenance.reference`);
        }
      });
    });
    requireCondition(solidCount > 0, "scene contains no renderable solids");

    const connections = requireArray(scene.connections, "scene.connections", false);
    const connectionIds = new Set();
    connections.forEach((connection, connectionIndex) => {
      const label = `scene.connections[${connectionIndex}]`;
      requireObject(connection, label);
      requireKeys(connection, ["id", "kind", "domain", "endpoints", "metadata"], ["joint"], label);
      const id = requireString(connection.id, `${label}.id`);
      requireCondition(!connectionIds.has(id), `duplicate connection id ${JSON.stringify(id)}`);
      connectionIds.add(id);
      requireCondition(["power", "signal", "attachment", "constraint"].includes(connection.kind),
        `${label}.kind is unsupported`);
      if (connection.domain !== null) requireString(connection.domain, `${label}.domain`);
      const connectionMetadata = requireObject(connection.metadata, `${label}.metadata`);
      requireCondition(Object.keys(connectionMetadata).length === 0,
        `${label}.metadata must be empty; physical semantics require typed fields`);
      const endpoints = requireArray(connection.endpoints, `${label}.endpoints`, true);
      requireCondition(endpoints.length >= 2, `${label}.endpoints must contain at least two endpoints`);
      endpoints.forEach((endpoint, endpointIndex) => {
        const endpointLabel = `${label}.endpoints[${endpointIndex}]`;
        requireObject(endpoint, endpointLabel);
        requireKeys(endpoint, ["component", "connector"], [], endpointLabel);
        const componentId = requireString(endpoint.component, `${endpointLabel}.component`);
        const connectorId = requireString(endpoint.connector, `${endpointLabel}.connector`);
        requireCondition(instanceIds.has(componentId),
          `${endpointLabel} references unknown component ${JSON.stringify(componentId)}`);
        requireCondition(connectorKeys.has(`${componentId}.${connectorId}`),
          `${endpointLabel} references unknown connector ${componentId}.${connectorId}`);
      });
      if (connection.kind === "attachment") {
        requireCondition(endpoints.length === 2, `${label} attachment must have two endpoints`);
        const joint = requireObject(connection.joint, `${label}.joint`);
        requireKeys(joint,
          ["kind", "behavior_binding", "coordinate", "zero_angle_rad", "coordinate_bindings"], [], `${label}.joint`);
        requireCondition(joint.kind === "fixed" || joint.kind === "revolute",
          `${label}.joint.kind is unsupported`);
        requireCondition(joint.behavior_binding === "kinematic_only" || joint.behavior_binding === "pmdl",
          `${label}.joint.behavior_binding is unsupported`);
        requireNumber(joint.zero_angle_rad, `${label}.joint.zero_angle_rad`, false);
        const bindings = requireArray(joint.coordinate_bindings,
          `${label}.joint.coordinate_bindings`, false);
        const bindingStates = new Set();
        bindings.forEach((binding, bindingIndex) => {
          const bindingLabel = `${label}.joint.coordinate_bindings[${bindingIndex}]`;
          requireObject(binding, bindingLabel);
          requireKeys(binding, ["state", "joint_angle_at_state_zero_rad"], [], bindingLabel);
          const state = requireString(binding.state, `${bindingLabel}.state`);
          requireCondition(!bindingStates.has(state), `${label}.joint repeats bound state ${state}`);
          bindingStates.add(state);
          requireNumber(binding.joint_angle_at_state_zero_rad,
            `${bindingLabel}.joint_angle_at_state_zero_rad`, false);
        });
        if (joint.kind === "revolute") {
          requireString(joint.coordinate, `${label}.joint.coordinate`);
          requireCondition(bindings.length > 0,
            `${label}.joint revolute joint requires coordinate_bindings`);
          requireCondition(bindings[0].state === joint.coordinate &&
            Math.abs(bindings[0].joint_angle_at_state_zero_rad - joint.zero_angle_rad) <= 1e-12,
          `${label}.joint primary coordinate/zero angle must match its first binding`);
        } else {
          requireCondition(joint.coordinate === null && joint.zero_angle_rad === 0 && bindings.length === 0,
            `${label}.joint fixed joint has invalid coordinate, zero angle, or bindings`);
        }
      } else {
        requireCondition(connection.joint === undefined || connection.joint === null,
          `${label} non-attachment cannot have a joint`);
      }
    });

    const staticPoses = requireObject(scene.body_poses, "scene.body_poses");
    const staticKeys = Object.keys(staticPoses);
    const staticMissing = Array.from(bodyKeys).filter((key) => !(key in staticPoses));
    const staticExtra = staticKeys.filter((key) => !bodyKeys.has(key));
    requireCondition(staticMissing.length === 0 && staticExtra.length === 0,
      `scene.body_poses does not exactly match bodies; missing=[${staticMissing.join(", ")}], unknown=[${staticExtra.join(", ")}]`);
    staticKeys.forEach((key) => requirePose(staticPoses[key], `scene.body_poses[${JSON.stringify(key)}]`));
    const staticConnectorPoses = requireObject(scene.connector_poses, "scene.connector_poses");
    const staticConnectorKeys = Object.keys(staticConnectorPoses);
    const connectorMissing = Array.from(spatialConnectorKeys).filter((key) => !(key in staticConnectorPoses));
    const connectorExtra = staticConnectorKeys.filter((key) => !spatialConnectorKeys.has(key));
    requireCondition(connectorMissing.length === 0 && connectorExtra.length === 0,
      `scene.connector_poses does not exactly match spatial connectors; missing=[${connectorMissing.join(", ")}], unknown=[${connectorExtra.join(", ")}]`);
    staticConnectorKeys.forEach((key) =>
      requirePose(staticConnectorPoses[key], `scene.connector_poses[${JSON.stringify(key)}]`));

    let frames = [{ time_s: 0, body_poses: staticPoses, connector_poses: staticConnectorPoses }];
    if (scene.body_pose_frames !== undefined) {
      const wrapper = requireObject(scene.body_pose_frames, "scene.body_pose_frames");
      requireKeys(wrapper, ["assembly_sha256", "frames"], [], "scene.body_pose_frames");
      requireCondition(HASH_PATTERN.test(wrapper.assembly_sha256),
        "scene.body_pose_frames.assembly_sha256 is not canonical");
      requireCondition(wrapper.assembly_sha256 === scene.assembly_sha256,
        `body pose frame assembly hash mismatch (${wrapper.assembly_sha256} != ${scene.assembly_sha256})`);
      frames = requireArray(wrapper.frames, "scene.body_pose_frames.frames", true);
    }
    let previousTime = -Infinity;
    frames.forEach((poseFrame, frameIndex) => {
      const frameLabel = `scene.body_pose_frames.frames[${frameIndex}]`;
      requireObject(poseFrame, frameLabel);
      requireKeys(poseFrame, ["time_s", "body_poses", "connector_poses"], [], frameLabel);
      const time = requireNumber(poseFrame.time_s, `${frameLabel}.time_s`, false);
      requireCondition(time >= 0 && time > previousTime,
        `${frameLabel}.time_s must be non-negative and increase strictly`);
      previousTime = time;
      const bodyPoses = requireObject(poseFrame.body_poses, `${frameLabel}.body_poses`);
      const poseKeys = Object.keys(bodyPoses);
      const missing = Array.from(bodyKeys).filter((key) => !(key in bodyPoses));
      const extra = poseKeys.filter((key) => !bodyKeys.has(key));
      requireCondition(missing.length === 0 && extra.length === 0,
        `${frameLabel}.body_poses does not exactly match bodies; missing=[${missing.join(", ")}], unknown=[${extra.join(", ")}]`);
      poseKeys.forEach((key) => requirePose(bodyPoses[key], `${frameLabel}.body_poses[${JSON.stringify(key)}]`));
      const connectorPoses = requireObject(poseFrame.connector_poses, `${frameLabel}.connector_poses`);
      const connectorPoseKeys = Object.keys(connectorPoses);
      const missingConnectors = Array.from(spatialConnectorKeys).filter((key) => !(key in connectorPoses));
      const extraConnectors = connectorPoseKeys.filter((key) => !spatialConnectorKeys.has(key));
      requireCondition(missingConnectors.length === 0 && extraConnectors.length === 0,
        `${frameLabel}.connector_poses does not exactly match spatial connectors; missing=[${missingConnectors.join(", ")}], unknown=[${extraConnectors.join(", ")}]`);
      connectorPoseKeys.forEach((key) =>
        requirePose(connectorPoses[key], `${frameLabel}.connector_poses[${JSON.stringify(key)}]`));
    });
    if (scene.body_pose_frames !== undefined) {
      requireCondition(JSON.stringify(frames[0].body_poses) === JSON.stringify(staticPoses),
        "scene.body_poses must equal the first hash-bound body pose frame");
      requireCondition(JSON.stringify(frames[0].connector_poses) === JSON.stringify(staticConnectorPoses),
        "scene.connector_poses must equal the first hash-bound connector pose frame");
    }
    return payload;
  }

  function colorFor(text) {
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return `hsl(${Math.abs(hash) % 360} 55% 57%)`;
  }

  function boxMesh(size) {
    const [x, y, z] = size.map((item) => item / 2);
    return {
      vertices: [
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
      ],
      faces: [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]],
    };
  }

  function sphereMesh(dimensions) {
    const radii = dimensions.map((item) => item / 2);
    const rings = 8, segments = 16;
    const vertices = [];
    for (let ring = 0; ring <= rings; ring += 1) {
      const latitude = Math.PI * ring / rings;
      for (let segment = 0; segment < segments; segment += 1) {
        const longitude = 2 * Math.PI * segment / segments;
        vertices.push([
          radii[0] * Math.sin(latitude) * Math.cos(longitude),
          radii[1] * Math.sin(latitude) * Math.sin(longitude),
          radii[2] * Math.cos(latitude),
        ]);
      }
    }
    const faces = [];
    for (let ring = 0; ring < rings; ring += 1) {
      for (let segment = 0; segment < segments; segment += 1) {
        const next = (segment + 1) % segments;
        faces.push([
          ring * segments + segment,
          ring * segments + next,
          (ring + 1) * segments + next,
          (ring + 1) * segments + segment,
        ]);
      }
    }
    return { vertices, faces };
  }

  function cylinderMesh(dimensions) {
    const segments = 20, half = dimensions[2] / 2;
    const radiusX = dimensions[0] / 2, radiusY = dimensions[1] / 2;
    const vertices = [];
    for (let ring = 0; ring < 2; ring += 1) {
      const z = ring ? half : -half;
      for (let segment = 0; segment < segments; segment += 1) {
        const angle = 2 * Math.PI * segment / segments;
        vertices.push([radiusX * Math.cos(angle), radiusY * Math.sin(angle), z]);
      }
    }
    const faces = [
      Array.from({ length: segments }, (_none, index) => index).reverse(),
      Array.from({ length: segments }, (_none, index) => segments + index),
    ];
    for (let segment = 0; segment < segments; segment += 1) {
      const next = (segment + 1) % segments;
      faces.push([segment, next, segments + next, segments + segment]);
    }
    return { vertices, faces };
  }

  function meshFor(geometry) {
    if (geometry.kind === "box") return boxMesh(geometry.dimensions_m);
    if (geometry.kind === "sphere") return sphereMesh(geometry.dimensions_m);
    if (geometry.kind === "cylinder") return cylinderMesh(geometry.dimensions_m);
    fail(`unsupported geometry kind ${JSON.stringify(geometry.kind)}`);
  }

  function rotateQuaternion(point, quaternion) {
    const [w, x, y, z] = quaternion;
    const [px, py, pz] = point;
    const tx = 2 * (y * pz - z * py);
    const ty = 2 * (z * px - x * pz);
    const tz = 2 * (x * py - y * px);
    return [
      px + w * tx + (y * tz - z * ty),
      py + w * ty + (z * tx - x * tz),
      pz + w * tz + (x * ty - y * tx),
    ];
  }

  function transformPose(point, pose) {
    const rotated = rotateQuaternion(point, pose.rotation_quaternion_wxyz);
    return rotated.map((value, index) => value + pose.translation_m[index]);
  }

  function rotateX(point, angle) {
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    return [point[0], point[1] * cosine - point[2] * sine, point[1] * sine + point[2] * cosine];
  }

  function rotateZ(point, angle) {
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    return [point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine, point[2]];
  }

  function createSlider(container, label, options, onInput) {
    const wrapper = document.createElement("label");
    wrapper.className = "slider-row";
    const text = document.createElement("span");
    text.textContent = label;
    const output = document.createElement("output");
    const input = document.createElement("input");
    input.type = "range";
    Object.entries(options).forEach(([key, value]) => { input[key] = String(value); });
    const update = () => {
      output.textContent = onInput(Number(input.value), false);
      render();
    };
    input.addEventListener("input", update);
    wrapper.append(text, output, input);
    container.append(wrapper);
    output.textContent = onInput(Number(input.value), true);
    return input;
  }

  function createCheckbox(container, label, checked, onInput) {
    const wrapper = document.createElement("label");
    wrapper.className = "slider-row checkbox-row";
    const text = document.createElement("span");
    text.textContent = label;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    const output = document.createElement("output");
    const update = (initial) => {
      output.textContent = input.checked ? "ON" : "OFF";
      onInput(input.checked, initial);
      if (!initial) render();
    };
    input.addEventListener("change", () => update(false));
    wrapper.append(text, output, input);
    container.append(wrapper);
    update(true);
    return input;
  }

  async function setupLiveControls(payload, container, statusText) {
    if (payload.live === undefined) return;
    const response = await fetch(payload.live.schema_endpoint, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`live control schema request failed with HTTP ${response.status}`);
    }
    const schema = await response.json();
    requireObject(schema, "live control schema");
    requireKeys(schema,
      ["schema", "assembly_sha256", "controller", "inputs", "values"], [], "live control schema");
    requireCondition(schema.schema === "contraption.live-controls/v1",
      "live control schema has an unsupported schema identifier");
    requireCondition(schema.assembly_sha256 === payload.assembly_sha256,
      "live control schema assembly hash differs from the rendered assembly");
    const controller = requireObject(schema.controller, "live control schema.controller");
    requireKeys(controller, ["id", "version", "sha256"], [], "live control schema.controller");
    requireString(controller.id, "live control schema.controller.id");
    requireString(controller.version, "live control schema.controller.version");
    requireCondition(HASH_PATTERN.test(controller.sha256),
      "live control schema.controller.sha256 is not canonical");
    const declarations = requireArray(schema.inputs, "live control schema.inputs", false);
    const values = requireObject(schema.values, "live control schema.values");
    const declaredNames = [];
    const inputs = new Map();

    const heading = document.createElement("p");
    heading.className = "panel-kicker live-control-heading";
    heading.textContent = `LIVE / ${controller.id}`;
    container.append(heading);

    declarations.forEach((declaration, index) => {
      const label = `live control schema.inputs[${index}]`;
      requireObject(declaration, label);
      requireKeys(declaration,
        ["name", "type", "default", "minimum", "maximum", "unit", "description"], [], label);
      const name = requireString(declaration.name, `${label}.name`);
      requireCondition(!declaredNames.includes(name), `duplicate live input ${JSON.stringify(name)}`);
      declaredNames.push(name);
      requireString(declaration.unit, `${label}.unit`);
      requireCondition(typeof declaration.description === "string", `${label}.description must be a string`);
      requireCondition(Object.prototype.hasOwnProperty.call(values, name),
        `live control schema.values is missing ${name}`);
      if (declaration.type === "boolean") {
        requireCondition(typeof declaration.default === "boolean" && typeof values[name] === "boolean",
          `${label} boolean default/value is invalid`);
        requireCondition(declaration.minimum === null && declaration.maximum === null,
          `${label} boolean bounds must be null`);
        inputs.set(name, createCheckbox(container, name, values[name], () => {}));
      } else {
        requireCondition(declaration.type === "number", `${label}.type is unsupported`);
        const minimum = requireNumber(declaration.minimum, `${label}.minimum`, false);
        const maximum = requireNumber(declaration.maximum, `${label}.maximum`, false);
        requireCondition(maximum > minimum, `${label} bounds must increase`);
        const defaultValue = requireNumber(declaration.default, `${label}.default`, false);
        const currentValue = requireNumber(values[name], `live control schema.values.${name}`, false);
        requireCondition(defaultValue >= minimum && defaultValue <= maximum,
          `${label}.default is outside its declared bounds`);
        requireCondition(currentValue >= minimum && currentValue <= maximum,
          `live control schema.values.${name} is outside its declared bounds`);
        const span = maximum - minimum;
        inputs.set(name, createSlider(container, `${name} (${declaration.unit})`, {
          min: minimum,
          max: maximum,
          step: Math.max(span / 400, 1e-9),
          value: values[name],
        }, (value) => value.toPrecision(5)));
      }
    });
    const extraValues = Object.keys(values).filter((name) => !declaredNames.includes(name));
    requireCondition(extraValues.length === 0,
      `live control schema.values contains unknown inputs: ${extraValues.join(", ")}`);

    const runButton = document.createElement("button");
    runButton.type = "button";
    runButton.className = "primary-button live-run-button";
    runButton.textContent = "Run canonical simulation";
    container.append(runButton);
    runButton.addEventListener("click", async () => {
      runButton.disabled = true;
      statusText.textContent = "Running canonical Python simulation…";
      try {
        const selected = Object.fromEntries(Array.from(inputs, ([name, input]) => [
          name,
          input.type === "checkbox" ? input.checked : Number(input.value),
        ]));
        const simulationResponse = await fetch(payload.live.simulate_endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          credentials: "same-origin",
          body: JSON.stringify({
            assembly_sha256: payload.assembly_sha256,
            inputs: selected,
          }),
        });
        const body = await simulationResponse.json();
        if (!simulationResponse.ok) {
          const detail = body && typeof body.error === "string" ? body.error : `HTTP ${simulationResponse.status}`;
          throw new Error(`canonical simulation rejected controls: ${detail}`);
        }
        validatedPayload({ ...payload, scene: body });
        statusText.textContent = "Canonical scene verified; reloading frames…";
        window.location.reload();
      } catch (error) {
        statusText.textContent = error instanceof Error ? error.message : String(error);
        runButton.disabled = false;
      }
    });
  }

  let render = () => {};

  function start() {
    const payload = validatedPayload();
    const scene = payload.scene;
    const canvas = byId("scene");
    const context = canvas.getContext("2d", { alpha: false });
    requireCondition(context !== null, "browser cannot create a two-dimensional canvas context");
    const timeline = byId("timeline");
    const playButton = byId("play");
    const tooltip = byId("component-tooltip");
    const statusText = byId("status-text");
    const frames = scene.body_pose_frames
      ? scene.body_pose_frames.frames
      : [{ time_s: 0, body_poses: scene.body_poses, connector_poses: scene.connector_poses }];
    timeline.max = String(frames.length - 1);

    const solids = [];
    const connectors = [];
    const connectorByKey = new Map();
    scene.components.forEach((component) => {
      component.bodies.forEach((body) => {
        body.solids.forEach((solid) => {
          const key = `${component.id}/${body.id}`;
          solids.push({
            component,
            body,
            solid,
            bodyKey: key,
            mesh: meshFor(solid.geometry),
            color: colorFor(`${key}/${solid.id}`),
          });
        });
      });
      component.connectors.forEach((connector) => {
        connectorByKey.set(`${component.id}.${connector.id}`, connector);
        if (connector.body !== null) {
          connectors.push({
            component,
            connector,
            poseKey: `${component.id}.${connector.id}`,
          });
        }
      });
    });

    const physicalComponents = scene.components.filter((component) => component.bodies.length > 0);
    const alphaValues = Object.fromEntries(physicalComponents.map((component) => [component.id, 1]));
    let globalAlpha = 1;
    let showConnectors = true;
    const controlsContainer = byId("external-controls");
    createCheckbox(controlsContainer, "Connector frames", true, (value, initial) => {
      showConnectors = value;
      if (!initial) statusText.textContent = `Connector overlays: ${value ? "ON" : "OFF"}`;
    });
    setupLiveControls(payload, controlsContainer, statusText).catch((error) => {
      statusText.textContent = error instanceof Error ? error.message : String(error);
    });

    const globalSlider = byId("global-alpha");
    globalSlider.addEventListener("input", () => {
      globalAlpha = Number(globalSlider.value);
      byId("global-alpha-value").textContent = `${Math.round(globalAlpha * 100)}%`;
      render();
    });
    const alphaContainer = byId("component-alpha");
    physicalComponents.forEach((component) => {
      createSlider(alphaContainer, component.id, {
        min: 0.03, max: 1, step: 0.01, value: 1,
      }, (value) => {
        alphaValues[component.id] = value;
        return `${Math.round(value * 100)}%`;
      });
    });

    const view = {
      yaw: -0.72, pitch: 0.54, zoom: 1, panX: 0, panY: 0,
      dragMode: null, pointerId: null, x: 0, y: 0,
    };
    const defaultView = Object.assign({}, view);
    let frame = 0;
    let playing = false;
    let speed = 1;
    let animationTime = frames[0].time_s;
    let lastAnimation = 0;
    let hitRegions = [];
    let connectorHitRegions = [];

    function canvasSize() {
      const rectangle = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(rectangle.width * ratio));
      const height = Math.max(1, Math.round(rectangle.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return { width, height, ratio };
    }

    function transformed(point) {
      return rotateX(rotateZ(point, view.yaw), view.pitch);
    }

    function solidWorld(item) {
      const bodyPose = frames[frame].body_poses[item.bodyKey];
      return {
        item,
        vertices: item.mesh.vertices.map((point) => transformPose(transformPose(point, item.solid.local_pose), bodyPose)),
        faces: item.mesh.faces,
      };
    }

    function connectorWorld(item, axisLength) {
      const connectorPose = frames[frame].connector_poses[item.poseKey];
      return {
        item,
        origin: connectorPose.translation_m.slice(),
        axis: transformPose([0, 0, axisLength], connectorPose),
      };
    }

    function rgba(color, alpha, shade) {
      const probe = document.createElement("canvas").getContext("2d");
      probe.fillStyle = color;
      const normalized = probe.fillStyle;
      if (normalized.startsWith("#")) {
        const hex = normalized.slice(1);
        const expanded = hex.length === 3 ? hex.split("").map((value) => value + value).join("") : hex;
        const values = [0, 2, 4].map((offset) => Math.max(0, Math.min(255,
          parseInt(expanded.slice(offset, offset + 2), 16) * shade)));
        return `rgba(${values[0]},${values[1]},${values[2]},${alpha})`;
      }
      return color;
    }

    function drawGrid(width, height, ratio) {
      context.fillStyle = "#081216";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "rgba(111, 157, 168, 0.08)";
      context.lineWidth = ratio;
      const spacing = 36 * ratio;
      for (let x = width % spacing; x < width; x += spacing) {
        context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
      }
      for (let y = height % spacing; y < height; y += spacing) {
        context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
      }
    }

    function connectorColor(domain) {
      if (domain === "electrical") return "#ffbd66";
      if (domain === "signal") return "#65a9ff";
      if (domain === "mechanical" || domain === "rigid_mechanical") return "#55e5c5";
      return "#eaf1f4";
    }

    render = function renderScene() {
      const { width, height, ratio } = canvasSize();
      drawGrid(width, height, ratio);
      const world = solids.map(solidWorld);
      const allPoints = world.flatMap((item) => item.vertices).map(transformed);
      requireCondition(allPoints.length > 0, "no geometry points are available for rendering");
      const bounds = allPoints.reduce((result, point) => ({
        minimum: result.minimum.map((value, index) => Math.min(value, point[index])),
        maximum: result.maximum.map((value, index) => Math.max(value, point[index])),
      }), { minimum: [Infinity, Infinity, Infinity], maximum: [-Infinity, -Infinity, -Infinity] });
      const center = bounds.minimum.map((value, index) => (value + bounds.maximum[index]) / 2);
      let extent = 0.01;
      allPoints.forEach((point) => {
        extent = Math.max(extent, Math.abs(point[0] - center[0]),
          Math.abs(point[1] - center[1]), Math.abs(point[2] - center[2]) * 0.6);
      });
      const scale = Math.min(width, height) * 0.39 * view.zoom / extent;
      const project = (point) => {
        const rotated = transformed(point);
        const relative = rotated.map((value, index) => value - center[index]);
        const perspective = 1 / Math.max(0.55, 1 + relative[2] / (extent * 5));
        return [
          width / 2 + view.panX * ratio + relative[0] * scale * perspective,
          height / 2 + view.panY * ratio - relative[1] * scale * perspective,
          relative[2],
        ];
      };
      const faces = [];
      world.forEach((worldSolid) => {
        const projected = worldSolid.vertices.map(project);
        worldSolid.faces.forEach((face, faceIndex) => {
          const points = face.map((index) => projected[index]);
          faces.push({
            item: worldSolid.item,
            points,
            depth: points.reduce((sum, point) => sum + point[2], 0) / points.length,
            shade: 0.72 + (faceIndex % 4) * 0.09,
          });
        });
      });
      faces.sort((left, right) => left.depth - right.depth);
      hitRegions = [];
      faces.forEach((face) => {
        const alpha = Math.max(0.01, Math.min(1,
          globalAlpha * alphaValues[face.item.component.id]));
        context.save();
        context.beginPath();
        context.moveTo(face.points[0][0], face.points[0][1]);
        face.points.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
        context.closePath();
        context.globalAlpha = alpha * 0.82;
        context.fillStyle = rgba(face.item.color, 1, face.shade);
        context.fill();
        context.globalAlpha = Math.min(0.7, alpha + 0.12);
        context.strokeStyle = "rgb(225,245,247)";
        context.lineWidth = Math.max(0.55, ratio * 0.65);
        context.stroke();
        context.restore();
        hitRegions.push({ item: face.item, points: face.points });
      });
      connectorHitRegions = [];
      if (showConnectors) {
        connectors.map((item) => connectorWorld(item, extent * 0.055)).forEach((worldConnector) => {
          const origin = project(worldConnector.origin);
          const axis = project(worldConnector.axis);
          const color = connectorColor(worldConnector.item.connector.domain);
          context.save();
          context.strokeStyle = color;
          context.fillStyle = "#081216";
          context.lineWidth = Math.max(1.25, ratio * 1.4);
          context.beginPath();
          context.moveTo(origin[0], origin[1]);
          context.lineTo(axis[0], axis[1]);
          context.stroke();
          context.beginPath();
          context.arc(origin[0], origin[1], Math.max(3.5, 3.5 * ratio), 0, Math.PI * 2);
          context.fill();
          context.stroke();
          context.restore();
          connectorHitRegions.push({
            item: worldConnector.item,
            x: origin[0],
            y: origin[1],
            radius: Math.max(7, 7 * ratio),
          });
        });
      }
      timeline.value = String(frame);
      byId("time-value").textContent = `${frames[frame].time_s.toFixed(3)} s`;
      byId("frame-value").textContent = `frame ${frame + 1} / ${frames.length}`;
    };

    function pointInPolygon(x, y, points) {
      let inside = false;
      for (let first = 0, second = points.length - 1; first < points.length; second = first++) {
        const xi = points[first][0], yi = points[first][1];
        const xj = points[second][0], yj = points[second][1];
        const intersects = ((yi > y) !== (yj > y)) &&
          x < (xj - xi) * (y - yi) / (yj - yi || 1e-12) + xi;
        if (intersects) inside = !inside;
      }
      return inside;
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 2) return;
      event.preventDefault();
      view.dragMode = event.button === 2 ? "rotate" : "pan";
      view.pointerId = event.pointerId;
      view.x = event.clientX;
      view.y = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (view.dragMode && event.pointerId === view.pointerId) {
        const deltaX = event.clientX - view.x, deltaY = event.clientY - view.y;
        if (view.dragMode === "rotate") {
          view.yaw += deltaX * 0.008;
          view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + deltaY * 0.008));
        } else {
          view.panX += deltaX;
          view.panY += deltaY;
        }
        view.x = event.clientX;
        view.y = event.clientY;
        tooltip.hidden = true;
        render();
        return;
      }
      const rectangle = canvas.getBoundingClientRect();
      const x = (event.clientX - rectangle.left) * canvas.width / rectangle.width;
      const y = (event.clientY - rectangle.top) * canvas.height / rectangle.height;
      const connectorHit = connectorHitRegions.slice().reverse().find((region) =>
        (x - region.x) * (x - region.x) + (y - region.y) * (y - region.y) <= region.radius * region.radius);
      const hit = hitRegions.slice().reverse().find((region) => pointInPolygon(x, y, region.points));
      if (connectorHit) {
        tooltip.hidden = false;
        tooltip.textContent = `${connectorHit.item.component.id}.${connectorHit.item.connector.id} \u00b7 ${connectorHit.item.connector.domain} \u00b7 ${connectorHit.item.connector.interface}`;
        tooltip.style.left = `${event.clientX - rectangle.left + 12}px`;
        tooltip.style.top = `${event.clientY - rectangle.top + 12}px`;
      } else if (hit) {
        tooltip.hidden = false;
        tooltip.textContent = `${hit.item.component.id} / ${hit.item.body.id} / ${hit.item.solid.id}`;
        tooltip.style.left = `${event.clientX - rectangle.left + 12}px`;
        tooltip.style.top = `${event.clientY - rectangle.top + 12}px`;
      } else {
        tooltip.hidden = true;
      }
    });
    function endDrag(event) {
      if (event.pointerId === view.pointerId) {
        view.dragMode = null;
        view.pointerId = null;
      }
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      view.zoom = Math.max(0.2, Math.min(8, view.zoom * Math.exp(-event.deltaY * 0.0012)));
      render();
    }, { passive: false });
    byId("reset-view").addEventListener("click", () => {
      Object.assign(view, defaultView);
      render();
    });

    function setFrame(value) {
      frame = Math.max(0, Math.min(frames.length - 1, Math.round(value)));
      animationTime = frames[frame].time_s;
      render();
    }
    timeline.addEventListener("input", () => setFrame(Number(timeline.value)));
    byId("step-back").addEventListener("click", () => { playing = false; setFrame(frame - 1); updatePlay(); });
    byId("step-forward").addEventListener("click", () => { playing = false; setFrame(frame + 1); updatePlay(); });
    byId("speed").addEventListener("change", (event) => { speed = Number(event.target.value); });
    function updatePlay() {
      playButton.textContent = playing ? "||" : "\u25b6";
      playButton.setAttribute("aria-label", playing ? "Pause pose frames" : "Play pose frames");
    }
    playButton.addEventListener("click", () => {
      playing = !playing;
      if (playing && frame >= frames.length - 1) setFrame(0);
      lastAnimation = performance.now();
      updatePlay();
      if (playing) requestAnimationFrame(animate);
    });
    function animate(timestamp) {
      if (!playing) return;
      const elapsed = Math.max(0, (timestamp - lastAnimation) / 1000) * speed;
      lastAnimation = timestamp;
      animationTime += elapsed;
      while (frame + 1 < frames.length && frames[frame + 1].time_s <= animationTime) frame += 1;
      if (frame >= frames.length - 1) {
        playing = false;
        updatePlay();
      }
      render();
      if (playing) requestAnimationFrame(animate);
    }

    function drawElectrical() {
      const svg = byId("electrical-diagram");
      const electrical = scene.connections.filter((item) => {
        const domains = item.endpoints.map((endpoint) =>
          connectorByKey.get(`${endpoint.component}.${endpoint.connector}`).domain);
        return item.kind === "signal" || item.domain === "electrical" ||
          domains.every((domain) => domain === "electrical" || domain === "signal");
      });
      const ids = Array.from(new Set(electrical.flatMap((item) =>
        item.endpoints.map((endpoint) => endpoint.component)))).sort();
      const width = Math.max(720, svg.clientWidth || 720), height = 220;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.replaceChildren();
      if (!electrical.length) {
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(width / 2));
        text.setAttribute("y", String(height / 2));
        text.setAttribute("class", "wire-label");
        text.textContent = "No electrical or signal connections in canonical scene";
        svg.append(text);
        byId("electrical-summary").textContent = "0 nets";
        return;
      }
      const positions = new Map(ids.map((id, index) => {
        const columns = Math.max(2, Math.ceil(Math.sqrt(ids.length * 2)));
        const rows = Math.ceil(ids.length / columns);
        return [id, {
          x: (index % columns + 0.5) * width / columns,
          y: (Math.floor(index / columns) + 0.5) * height / rows,
        }];
      }));
      electrical.forEach((connection) => {
        connection.endpoints.slice(1).forEach((destination) => {
          const source = connection.endpoints[0];
          const first = positions.get(source.component), second = positions.get(destination.component);
          const bend = (first.x + second.x) / 2;
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("d", `M ${first.x} ${first.y} C ${bend} ${first.y}, ${bend} ${second.y}, ${second.x} ${second.y}`);
          path.setAttribute("class", `wire ${connection.kind === "signal" ? "signal" : "power"}`);
          svg.append(path);
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.setAttribute("x", String(bend));
          label.setAttribute("y", String((first.y + second.y) / 2 - 5));
          label.setAttribute("class", "wire-label");
          label.textContent = `${source.connector} \u2192 ${destination.connector}`;
          svg.append(label);
        });
      });
      ids.forEach((id) => {
        const position = positions.get(id);
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", "electrical-node");
        const rectangle = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rectangle.setAttribute("x", String(position.x - 62));
        rectangle.setAttribute("y", String(position.y - 25));
        rectangle.setAttribute("width", "124");
        rectangle.setAttribute("height", "50");
        const name = document.createElementNS("http://www.w3.org/2000/svg", "text");
        name.setAttribute("x", String(position.x));
        name.setAttribute("y", String(position.y - 2));
        name.textContent = id;
        const count = document.createElementNS("http://www.w3.org/2000/svg", "text");
        count.setAttribute("x", String(position.x));
        count.setAttribute("y", String(position.y + 13));
        count.setAttribute("class", "sub");
        count.textContent = `${electrical.filter((item) => item.endpoints.some((endpoint) => endpoint.component === id)).length} connection(s)`;
        group.append(rectangle, name, count);
        svg.append(group);
      });
      byId("electrical-summary").textContent = `${ids.length} nodes \u00b7 ${electrical.length} nets`;
    }

    byId("model-summary").textContent = `${scene.components.length} components \u00b7 ${solids.length} solids \u00b7 ${connectors.length} spatial connectors \u00b7 ${frames.length} pose frames`;
    byId("assembly-hash").textContent = scene.assembly_sha256;
    statusText.textContent = `Canonical assembly verified \u00b7 ${scene.assembly_sha256.slice(7, 19)}`;
    drawElectrical();
    render();
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => { drawElectrical(); render(); }).observe(canvas.parentElement);
    } else {
      window.addEventListener("resize", () => { drawElectrical(); render(); });
    }
  }

  function reportFatal(error) {
    const message = error instanceof Error ? error.message : String(error);
    const status = byId("status-text");
    if (status) status.textContent = message;
    const statusDot = document.querySelector(".status-dot");
    if (statusDot) statusDot.style.background = "#ff5f6d";
    const canvas = byId("scene");
    if (canvas) {
      const context = canvas.getContext("2d");
      const rectangle = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rectangle.width));
      canvas.height = Math.max(1, Math.round(rectangle.height));
      context.fillStyle = "#081216";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#ff8b94";
      context.font = "16px system-ui";
      context.textAlign = "center";
      context.fillText("Viewer rejected incomplete or mismatched assembly data", canvas.width / 2, canvas.height / 2);
    }
  }

  try {
    start();
  } catch (error) {
    reportFatal(error);
    setTimeout(() => { throw error; }, 0);
  }
}());
