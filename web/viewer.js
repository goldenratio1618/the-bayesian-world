(function () {
  "use strict";

  const payloadNode = document.getElementById("contraption-data");
  const payload = JSON.parse(payloadNode.textContent);
  const spec = payload.specification || {};
  const simulation = payload.simulation || {};
  const runtime = payload.runtime || null;

  const byId = (id) => document.getElementById(id);
  const canvas = byId("scene");
  const context = canvas.getContext("2d", { alpha: false });
  const timeline = byId("timeline");
  const playButton = byId("play");
  const tooltip = byId("component-tooltip");
  const statusText = byId("status-text");

  function objectList(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") {
      return Object.keys(value).sort().map((key) => Object.assign({ id: key }, value[key]));
    }
    return [];
  }

  function componentId(component, index) {
    return String(component.id || component.name || `component_${index}`);
  }

  function metadata(item) {
    return item && item.metadata && typeof item.metadata === "object" ? item.metadata : {};
  }

  function finite(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function firstDefined() {
    for (let index = 0; index < arguments.length; index += 1) {
      if (arguments[index] !== undefined && arguments[index] !== null) return arguments[index];
    }
    return undefined;
  }

  function vector(value, fallback) {
    if (Array.isArray(value) && value.length >= 3) {
      return [finite(value[0], fallback[0]), finite(value[1], fallback[1]), finite(value[2], fallback[2])];
    }
    if (value && typeof value === "object") {
      return [finite(value.x, fallback[0]), finite(value.y, fallback[1]), finite(value.z, fallback[2])];
    }
    return fallback.slice();
  }

  function meanValue(value, fallback) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      value = firstDefined(value.mean, value.value, value.nominal);
    }
    return finite(value, fallback);
  }

  const rawComponents = objectList(spec.components);
  const components = rawComponents.map((component, index) => {
    const geometry = component.geometry && typeof component.geometry === "object" ? component.geometry : {};
    const geometryMeta = geometry.metadata && typeof geometry.metadata === "object" ? geometry.metadata : {};
    const meta = metadata(component);
    const dimensions = vector(geometry.dimensions || geometry.size, [0.01, 0.01, 0.01]).map((item) => Math.max(Math.abs(item), 1e-6));
    const position = vector(geometry.position || geometry.translation || geometryMeta.translation_m || geometryMeta.position || meta.position || meta.placement, [0, 0, 0]);
    const rotation = vector(geometry.rotation || geometry.orientation || geometryMeta.rotation_rpy_rad || geometryMeta.rotation || meta.rotation, [0, 0, 0]);
    return {
      raw: component,
      id: componentId(component, index),
      label: String(meta.display_name || component.name || componentId(component, index)),
      model: typeof component.model === "string" ? component.model : String(component.category || "component"),
      geometry,
      dimensions,
      position,
      rotation,
      color: String(geometry.color || geometryMeta.color || meta.color || colorFor(componentId(component, index))),
      fixed: Boolean(meta.fixed || meta.world_fixed || geometry.frame === "world"),
      alpha: 1,
    };
  });
  const componentMap = new Map(components.map((item) => [item.id, item]));

  function colorFor(text) {
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue} 55% 57%)`;
  }

  function endpoints(connection) {
    let values = connection.endpoints;
    if (!Array.isArray(values)) {
      if (connection.from || connection.to) values = [connection.from, connection.to];
      else if (connection.source || connection.target) values = [connection.source, connection.target];
      else if (connection.a || connection.b) values = [connection.a, connection.b];
      else values = [];
    }
    return values.map((endpoint) => {
      if (typeof endpoint === "string") {
        const pieces = endpoint.split(".");
        return { component: pieces.shift() || "?", port: pieces.join(".") || "?" };
      }
      endpoint = endpoint || {};
      return {
        component: String(endpoint.component || endpoint.component_id || "?"),
        port: String(endpoint.port || endpoint.port_name || "?"),
      };
    });
  }

  function connectionKind(connection) {
    const meta = metadata(connection);
    const kind = String(connection.kind || connection.domain || meta.domain || "unknown").toLowerCase();
    if (["power", "electrical", "wire"].includes(kind) || String(meta.domain).toLowerCase() === "electrical") return "power";
    if (["signal", "measurement", "command", "data"].includes(kind)) return "signal";
    return kind;
  }

  const connections = objectList(spec.connections).map((connection, index) => ({
    raw: connection,
    id: String(connection.id || connection.name || `connection_${index}`),
    kind: connectionKind(connection),
    endpoints: endpoints(connection),
  }));

  function simulationData(source) {
    let times = source.time || source.times || source.t || [];
    if (!Array.isArray(times) && times && typeof times.length === "number") times = Array.from(times);
    times = Array.isArray(times) ? times.map((value) => finite(value, 0)) : [];
    const names = source.state_names || source.states || [];
    let means = source.mean || source.state_mean || (source.summary && source.summary.mean);
    if (!Array.isArray(means) && Array.isArray(source.samples)) {
      const samples = source.samples;
      if (samples.length && Array.isArray(samples[0]) && Array.isArray(samples[0][0])) {
        const count = samples.length;
        means = samples[0].map((row, timeIndex) => row.map((_value, stateIndex) => {
          let total = 0;
          for (let sampleIndex = 0; sampleIndex < count; sampleIndex += 1) {
            total += finite(samples[sampleIndex][timeIndex][stateIndex], 0);
          }
          return total / count;
        }));
      } else {
        means = samples;
      }
    }
    if (means && !Array.isArray(means) && typeof means === "object") {
      const mapped = names.map((name) => means[name]);
      const length = Math.max(0, ...mapped.map((series) => Array.isArray(series) ? series.length : 0));
      means = Array.from({ length }, (_none, timeIndex) => mapped.map((series) => finite(series && series[timeIndex], 0)));
    }
    means = Array.isArray(means) ? means : [];
    if (!times.length && means.length) times = means.map((_row, index) => index);
    if (!times.length) times = [0];
    return {
      times,
      names: Array.isArray(names) ? names.map(String) : [],
      means,
      components: source.components && typeof source.components === "object" ? source.components : {},
    };
  }

  const trajectory = simulationData(simulation);
  timeline.max = String(Math.max(0, trajectory.times.length - 1));

  const stateIndex = new Map(trajectory.names.map((name, index) => [name, index]));
  function trajectoryState(name, frame) {
    const index = stateIndex.get(name);
    if (index === undefined || !trajectory.means[frame]) return undefined;
    return finite(trajectory.means[frame][index], undefined);
  }

  function declaredControls() {
    const meta = metadata(spec);
    let values = spec.external_controls || meta.external_controls || [];
    values = objectList(values);
    if (!values.length) {
      values = objectList(spec.controls).filter((control) => {
        const source = typeof control.source === "string" ? control.source : (control.source && control.source.kind);
        return control.external === true || String(source || "").toLowerCase().startsWith("external");
      });
    }
    return values.flatMap((control, index) => {
      const settings = control.settings && typeof control.settings === "object" ? control.settings : {};
      const source = typeof control.source === "string" ? control.source : "";
      const sourceTail = source.split(":").pop().split(".").pop();
      const name = String(control.name || control.signal || sourceTail || `control_${index}`);
      const label = String(control.label || name.split("_").join(" "));
      const unit = String(firstDefined(control.unit, settings.unit, "1"));
      const type = String(firstDefined(control.type, settings.type, "number"));
      const effects = objectList(control.visual_effects || control.effects || control.bindings || settings.visual_effects);
      const shape = firstDefined(control.shape, settings.shape);
      const defaultValue = firstDefined(control.default, control.value, settings.default);
      if (Array.isArray(shape) && shape.length === 1 && Number(shape[0]) > 1) {
        const count = Math.min(8, Math.floor(Number(shape[0])));
        const defaults = Array.isArray(defaultValue) ? defaultValue : [];
        const axes = ["x", "y", "z", "w"];
        return Array.from({ length: count }, (_none, axisIndex) => ({
          raw: control,
          name: `${name}_${axes[axisIndex] || axisIndex}`,
          label: `${label} ${String(axes[axisIndex] || axisIndex).toUpperCase()}`,
          unit,
          type: "number",
          min: meanValue(firstDefined(control.minimum, control.min, settings.minimum, settings.min), -2),
          max: meanValue(firstDefined(control.maximum, control.max, settings.maximum, settings.max), 2),
          step: meanValue(firstDefined(control.step, settings.step), 0.01),
          value: meanValue(defaults[axisIndex], 0),
          effects,
        }));
      }
      const min = meanValue(firstDefined(control.minimum, control.min, settings.minimum, settings.min), -1);
      const max = meanValue(firstDefined(control.maximum, control.max, settings.maximum, settings.max), 1);
      return [{
        raw: control,
        name,
        label,
        unit,
        type,
        min,
        max: max > min ? max : min + 1,
        step: meanValue(firstDefined(control.step, settings.step), Math.max((max - min) / 100, 0.001)),
        value: type === "boolean" ? Boolean(defaultValue) : meanValue(defaultValue, Math.min(max, Math.max(min, 0))),
        effects,
      }];
    }).sort((left, right) => left.name.localeCompare(right.name));
  }

  const controls = declaredControls();
  const controlValues = Object.fromEntries(controls.map((item) => [item.name, item.value]));
  const alphaValues = Object.fromEntries(components.map((item) => [item.id, 1]));
  let globalAlpha = 1;

  function formatNumber(value, unit) {
    const amount = Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(3);
    return unit && unit !== "1" ? `${amount} ${unit}` : amount;
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
      render();
    };
    input.addEventListener("change", () => update(false));
    wrapper.append(text, output, input);
    container.append(wrapper);
    update(true);
    return input;
  }

  const controlsContainer = byId("external-controls");
  if (controls.length) {
    controls.forEach((control) => {
      if (control.type === "boolean") {
        createCheckbox(controlsContainer, control.label, control.value, (value, initial) => {
          controlValues[control.name] = value;
          if (!initial) statusText.textContent = `${control.label}: ${value ? "ON" : "OFF"}`;
        });
      } else {
        createSlider(controlsContainer, control.label, {
          min: control.min, max: control.max, step: control.step, value: control.value,
        }, (value, initial) => {
          controlValues[control.name] = value;
          if (!initial) statusText.textContent = `${control.label}: ${formatNumber(value, control.unit)}`;
          return formatNumber(value, control.unit);
        });
      }
    });
  } else {
    controlsContainer.innerHTML = '<p class="empty-state">No external controls are declared in this specification.</p>';
  }

  const globalSlider = byId("global-alpha");
  globalSlider.addEventListener("input", () => {
    globalAlpha = Number(globalSlider.value);
    byId("global-alpha-value").textContent = `${Math.round(globalAlpha * 100)}%`;
    render();
  });
  const alphaContainer = byId("component-alpha");
  components.forEach((component) => {
    createSlider(alphaContainer, component.label, { min: 0.03, max: 1, step: 0.01, value: 1 }, (value) => {
      alphaValues[component.id] = value;
      return `${Math.round(value * 100)}%`;
    });
  });

  const view = { yaw: -0.72, pitch: 0.54, zoom: 1, dragging: false, x: 0, y: 0 };
  const defaultView = Object.assign({}, view);
  let frame = 0;
  let playing = false;
  let speed = 1;
  let animationTime = 0;
  let lastAnimation = 0;
  let hitRegions = [];

  function rotateX(point, angle) {
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    return [point[0], point[1] * cosine - point[2] * sine, point[1] * sine + point[2] * cosine];
  }
  function rotateY(point, angle) {
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    return [point[0] * cosine + point[2] * sine, point[1], -point[0] * sine + point[2] * cosine];
  }
  function rotateZ(point, angle) {
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    return [point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine, point[2]];
  }
  function rotateEuler(point, rotation) {
    return rotateZ(rotateY(rotateX(point, rotation[0]), rotation[1]), rotation[2]);
  }

  function boxMesh(dimensions) {
    const [x, y, z] = dimensions.map((value) => value / 2);
    return {
      vertices: [[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]],
      faces: [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
    };
  }

  function cylinderMesh(dimensions) {
    const radius = Math.max(dimensions[0], dimensions[1]) / 2;
    const half = dimensions[2] / 2;
    const vertices = [], faces = [];
    const segments = 16;
    for (let side = -1; side <= 1; side += 2) {
      for (let index = 0; index < segments; index += 1) {
        const angle = index * Math.PI * 2 / segments;
        vertices.push([radius * Math.cos(angle), radius * Math.sin(angle), side * half]);
      }
    }
    faces.push(Array.from({ length: segments }, (_none, index) => index).reverse());
    faces.push(Array.from({ length: segments }, (_none, index) => index + segments));
    for (let index = 0; index < segments; index += 1) {
      const next = (index + 1) % segments;
      faces.push([index, next, next + segments, index + segments]);
    }
    return { vertices, faces };
  }

  function sphereMesh(dimensions) {
    const radii = dimensions.map((value) => value / 2);
    const vertices = [], faces = [];
    const rings = 8, segments = 14;
    for (let ring = 0; ring <= rings; ring += 1) {
      const latitude = -Math.PI / 2 + ring * Math.PI / rings;
      for (let segment = 0; segment < segments; segment += 1) {
        const longitude = segment * Math.PI * 2 / segments;
        vertices.push([
          radii[0] * Math.cos(latitude) * Math.cos(longitude),
          radii[1] * Math.cos(latitude) * Math.sin(longitude),
          radii[2] * Math.sin(latitude),
        ]);
      }
    }
    for (let ring = 0; ring < rings; ring += 1) {
      for (let segment = 0; segment < segments; segment += 1) {
        const next = (segment + 1) % segments;
        const first = ring * segments + segment;
        const second = ring * segments + next;
        faces.push([first, second, second + segments, first + segments]);
      }
    }
    return { vertices, faces };
  }

  function meshFor(component) {
    const geometry = component.geometry;
    if (Array.isArray(geometry.vertices) && Array.isArray(geometry.faces)) {
      const vertices = geometry.vertices.map((point) => vector(point, [0, 0, 0]));
      const faces = geometry.faces.filter(Array.isArray).map((face) => face.map(Number).filter((index) => Number.isInteger(index) && index >= 0 && index < vertices.length));
      if (vertices.length >= 3 && faces.some((face) => face.length >= 3)) return { vertices, faces };
    }
    const kind = String(geometry.kind || geometry.shape || "box").toLowerCase();
    if (["cylinder", "wheel", "shaft"].includes(kind)) return cylinderMesh(component.dimensions);
    if (["sphere", "ellipsoid", "ball"].includes(kind)) return sphereMesh(component.dimensions);
    return boxMesh(component.dimensions);
  }

  function visualBindings() {
    const meta = metadata(spec);
    const visualization = spec.visualization && typeof spec.visualization === "object" ? spec.visualization : {};
    return objectList(visualization.bindings || meta.visual_bindings || spec.visual_bindings);
  }
  const bindings = visualBindings();

  function runtimeStates(targetTime) {
    if (!runtime || !Array.isArray(runtime.A) || !Array.isArray(runtime.B)) return null;
    const names = runtime.state_names || runtime.states || [];
    const inputNames = runtime.input_names || runtime.inputs || [];
    const state = (runtime.initial_state || runtime.x0 || names.map(() => 0)).map((value) => finite(value, 0));
    const bias = runtime.dynamics_bias || runtime.bias || state.map(() => 0);
    const nominal = Math.max(1e-5, finite(runtime.nominal_dt || runtime.dt, 0.01));
    const steps = Math.min(20000, Math.max(0, Math.ceil(targetTime / nominal)));
    const dt = steps ? targetTime / steps : nominal;
    for (let tick = 0; tick < steps; tick += 1) {
      const derivative = state.map((value, row) => {
        let total = finite(bias[row], 0);
        const matrixRow = runtime.A[row] || [];
        for (let column = 0; column < state.length; column += 1) total += finite(matrixRow[column], 0) * state[column];
        const inputRow = runtime.B[row] || [];
        for (let column = 0; column < inputNames.length; column += 1) total += finite(inputRow[column], 0) * finite(controlValues[inputNames[column]], 0);
        return total;
      });
      for (let index = 0; index < state.length; index += 1) state[index] += dt * derivative[index];
    }
    return Object.fromEntries(names.map((name, index) => [String(name), state[index]]));
  }

  function setPoseProperty(pose, property, value, additive) {
    const normalized = String(property || "").replace("translation", "position").replace("orientation", "rotation");
    const [group, axis] = normalized.split(".");
    const index = { x: 0, y: 1, z: 2, roll: 0, pitch: 1, yaw: 2 }[axis];
    if (!(group in pose) || index === undefined) return;
    pose[group][index] = additive ? pose[group][index] + value : value;
  }

  function componentPose(component, currentFrame) {
    const pose = { position: component.position.slice(), rotation: component.rotation.slice() };
    const time = trajectory.times[currentFrame] || 0;
    const liveStates = runtimeStates(time);
    const stateValue = (name) => liveStates && name in liveStates ? liveStates[name] : trajectoryState(name, currentFrame);

    const direct = trajectory.components[component.id];
    if (direct && typeof direct === "object") {
      const positionSeries = direct.position || direct.translation;
      const rotationSeries = direct.rotation || direct.orientation;
      if (Array.isArray(positionSeries) && positionSeries[currentFrame]) pose.position = vector(positionSeries[currentFrame], pose.position);
      if (Array.isArray(rotationSeries) && rotationSeries[currentFrame]) pose.rotation = vector(rotationSeries[currentFrame], pose.rotation);
    }

    if (bindings.length) {
      bindings.filter((binding) => String(binding.component || binding.component_id) === component.id).forEach((binding) => {
        const value = stateValue(String(binding.state || binding.signal));
        if (value !== undefined) setPoseProperty(pose, binding.property, finite(binding.offset, 0) + finite(binding.scale, 1) * value, Boolean(binding.additive));
      });
    } else if (!component.fixed) {
      const x = stateValue("x"), y = stateValue("y"), yaw = stateValue("yaw");
      if (yaw !== undefined) {
        pose.position = rotateZ(pose.position, yaw);
        pose.rotation[2] += yaw;
      }
      if (x !== undefined) pose.position[0] += x;
      if (y !== undefined) pose.position[1] += y;
      const arm = stateValue("arm_elevation");
      if (arm !== undefined && component.id.toLowerCase().includes("arm")) pose.position[2] += arm;
      const camera = stateValue("camera_pitch");
      if (camera !== undefined && component.id.toLowerCase().includes("camera")) pose.rotation[1] += camera;
    }

    controls.forEach((control) => {
      control.effects.filter((effect) => String(effect.component || effect.component_id) === component.id).forEach((effect) => {
        const value = finite(effect.offset, 0) + finite(firstDefined(effect.scale, effect.gain), 1) * controlValues[control.name];
        setPoseProperty(pose, effect.property, value, effect.additive !== false);
      });
    });
    return pose;
  }

  function componentWorld(component, currentFrame) {
    const pose = componentPose(component, currentFrame);
    const mesh = meshFor(component);
    return {
      component,
      vertices: mesh.vertices.map((point) => {
        const local = rotateEuler(point, pose.rotation);
        return [local[0] + pose.position[0], local[1] + pose.position[1], local[2] + pose.position[2]];
      }),
      faces: mesh.faces,
    };
  }

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

  function rgba(color, alpha, shade) {
    const probe = document.createElement("canvas").getContext("2d");
    probe.fillStyle = color;
    const normalized = probe.fillStyle;
    if (normalized.startsWith("#")) {
      const hex = normalized.slice(1);
      const expanded = hex.length === 3 ? hex.split("").map((value) => value + value).join("") : hex;
      const values = [0, 2, 4].map((offset) => Math.max(0, Math.min(255, parseInt(expanded.slice(offset, offset + 2), 16) * shade)));
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

  function render() {
    const { width, height, ratio } = canvasSize();
    drawGrid(width, height, ratio);
    const world = components.map((component) => componentWorld(component, frame));
    const allPoints = world.flatMap((item) => item.vertices).map(transformed);
    let extent = 0.01;
    allPoints.forEach((point) => { extent = Math.max(extent, Math.abs(point[0]), Math.abs(point[1]), Math.abs(point[2]) * 0.6); });
    const scale = Math.min(width, height) * 0.39 * view.zoom / extent;
    const project = (point) => {
      const rotated = transformed(point);
      const perspective = 1 / Math.max(0.55, 1 + rotated[2] / (extent * 5));
      return [width / 2 + rotated[0] * scale * perspective, height / 2 - rotated[1] * scale * perspective, rotated[2]];
    };
    const faces = [];
    world.forEach((item) => {
      const projected = item.vertices.map(project);
      item.faces.forEach((face, faceIndex) => {
        const points = face.map((index) => projected[index]).filter(Boolean);
        if (points.length >= 3) {
          faces.push({
            component: item.component,
            points,
            depth: points.reduce((sum, point) => sum + point[2], 0) / points.length,
            shade: 0.72 + (faceIndex % 4) * 0.09,
          });
        }
      });
    });
    faces.sort((left, right) => left.depth - right.depth);
    hitRegions = [];
    faces.forEach((face) => {
      const alpha = Math.max(0.01, Math.min(1, globalAlpha * alphaValues[face.component.id]));
      context.save();
      context.beginPath();
      context.moveTo(face.points[0][0], face.points[0][1]);
      face.points.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
      context.closePath();
      context.globalAlpha = alpha * 0.82;
      context.fillStyle = rgba(face.component.color, 1, face.shade);
      context.fill();
      context.globalAlpha = Math.min(0.7, alpha + 0.12);
      context.strokeStyle = "rgb(225,245,247)";
      context.lineWidth = Math.max(0.55, ratio * 0.65);
      context.stroke();
      context.restore();
      hitRegions.push({ component: face.component, points: face.points });
    });
    if (!components.length) {
      context.fillStyle = "#8da1aa";
      context.font = `${14 * ratio}px system-ui`;
      context.textAlign = "center";
      context.fillText("No component geometry declared", width / 2, height / 2);
    }
    timeline.value = String(frame);
    byId("time-value").textContent = `${finite(trajectory.times[frame], 0).toFixed(3)} s`;
    byId("frame-value").textContent = `frame ${frame + 1} / ${trajectory.times.length}`;
  }

  function pointInPolygon(x, y, points) {
    let inside = false;
    for (let first = 0, second = points.length - 1; first < points.length; second = first++) {
      const xi = points[first][0], yi = points[first][1], xj = points[second][0], yj = points[second][1];
      const intersects = ((yi > y) !== (yj > y)) && x < (xj - xi) * (y - yi) / (yj - yi || 1e-12) + xi;
      if (intersects) inside = !inside;
    }
    return inside;
  }

  canvas.addEventListener("pointerdown", (event) => {
    view.dragging = true; view.x = event.clientX; view.y = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (view.dragging) {
      view.yaw += (event.clientX - view.x) * 0.008;
      view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (event.clientY - view.y) * 0.008));
      view.x = event.clientX; view.y = event.clientY;
      tooltip.hidden = true;
      render();
      return;
    }
    const rectangle = canvas.getBoundingClientRect();
    const ratioX = canvas.width / rectangle.width, ratioY = canvas.height / rectangle.height;
    const x = (event.clientX - rectangle.left) * ratioX, y = (event.clientY - rectangle.top) * ratioY;
    const hit = hitRegions.slice().reverse().find((region) => pointInPolygon(x, y, region.points));
    if (hit) {
      tooltip.hidden = false;
      tooltip.textContent = `${hit.component.label} \u00b7 ${hit.component.model}`;
      tooltip.style.left = `${event.clientX - rectangle.left + 12}px`;
      tooltip.style.top = `${event.clientY - rectangle.top + 12}px`;
    } else tooltip.hidden = true;
  });
  canvas.addEventListener("pointerup", () => { view.dragging = false; });
  canvas.addEventListener("pointercancel", () => { view.dragging = false; });
  canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
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
    frame = Math.max(0, Math.min(trajectory.times.length - 1, Math.round(value)));
    animationTime = trajectory.times[frame];
    render();
  }
  timeline.addEventListener("input", () => setFrame(Number(timeline.value)));
  byId("step-back").addEventListener("click", () => { playing = false; setFrame(frame - 1); updatePlay(); });
  byId("step-forward").addEventListener("click", () => { playing = false; setFrame(frame + 1); updatePlay(); });
  byId("speed").addEventListener("change", (event) => { speed = Number(event.target.value); });
  function updatePlay() {
    playButton.textContent = playing ? "||" : "\u25b6";
    playButton.setAttribute("aria-label", playing ? "Pause simulation" : "Play simulation");
  }
  playButton.addEventListener("click", () => {
    playing = !playing;
    if (playing && frame >= trajectory.times.length - 1) setFrame(0);
    lastAnimation = performance.now();
    updatePlay();
    if (playing) requestAnimationFrame(animate);
  });
  function animate(timestamp) {
    if (!playing) return;
    const elapsed = Math.max(0, (timestamp - lastAnimation) / 1000) * speed;
    lastAnimation = timestamp;
    animationTime += elapsed;
    while (frame + 1 < trajectory.times.length && trajectory.times[frame + 1] <= animationTime) frame += 1;
    if (frame >= trajectory.times.length - 1) {
      playing = false;
      updatePlay();
    }
    render();
    if (playing) requestAnimationFrame(animate);
  }

  function drawElectrical() {
    const svg = byId("electrical-diagram");
    const electrical = connections.filter((item) => ["power", "signal"].includes(item.kind) && item.endpoints.length >= 2);
    const ids = Array.from(new Set(electrical.flatMap((item) => item.endpoints.map((endpoint) => endpoint.component)))).sort();
    const width = Math.max(720, svg.clientWidth || 720), height = 220;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    if (!electrical.length) {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(width / 2)); text.setAttribute("y", String(height / 2));
      text.setAttribute("class", "wire-label"); text.textContent = "No electrical connections declared";
      svg.append(text);
      byId("electrical-summary").textContent = "0 nets";
      return;
    }
    const positions = new Map(ids.map((id, index) => {
      const columns = Math.max(2, Math.ceil(Math.sqrt(ids.length * 2)));
      const rows = Math.ceil(ids.length / columns);
      const column = index % columns, row = Math.floor(index / columns);
      return [id, { x: (column + 0.5) * width / columns, y: (row + 0.5) * height / rows }];
    }));
    electrical.forEach((connection) => {
      connection.endpoints.slice(1).forEach((destination) => {
        const firstEndpoint = connection.endpoints[0];
        const first = positions.get(firstEndpoint.component), second = positions.get(destination.component);
        if (!first || !second) return;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const bend = (first.x + second.x) / 2;
        path.setAttribute("d", `M ${first.x} ${first.y} C ${bend} ${first.y}, ${bend} ${second.y}, ${second.x} ${second.y}`);
        path.setAttribute("class", `wire ${connection.kind}`);
        svg.append(path);
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(bend)); label.setAttribute("y", String((first.y + second.y) / 2 - 5));
        label.setAttribute("class", "wire-label");
        label.textContent = `${firstEndpoint.port} \u2192 ${destination.port}`;
        svg.append(label);
      });
    });
    ids.forEach((id) => {
      const position = positions.get(id);
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", "electrical-node");
      const rectangle = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rectangle.setAttribute("x", String(position.x - 62)); rectangle.setAttribute("y", String(position.y - 25));
      rectangle.setAttribute("width", "124"); rectangle.setAttribute("height", "50");
      const name = document.createElementNS("http://www.w3.org/2000/svg", "text");
      name.setAttribute("x", String(position.x)); name.setAttribute("y", String(position.y - 2)); name.textContent = id;
      const count = document.createElementNS("http://www.w3.org/2000/svg", "text");
      count.setAttribute("x", String(position.x)); count.setAttribute("y", String(position.y + 13)); count.setAttribute("class", "sub");
      count.textContent = `${electrical.filter((item) => item.endpoints.some((endpoint) => endpoint.component === id)).length} connection(s)`;
      group.append(rectangle, name, count); svg.append(group);
    });
    byId("electrical-summary").textContent = `${ids.length} nodes \u00b7 ${electrical.length} nets`;
  }

  byId("model-summary").textContent = `${components.length} components \u00b7 ${connections.length} connections \u00b7 ${trajectory.times.length} frames`;
  statusText.textContent = runtime ? "Offline live model ready" : trajectory.times.length > 1 ? "Offline trajectory ready" : "Offline geometry ready";
  drawElectrical();
  render();
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => { drawElectrical(); render(); }).observe(canvas.parentElement);
  } else window.addEventListener("resize", () => { drawElectrical(); render(); });
}());
