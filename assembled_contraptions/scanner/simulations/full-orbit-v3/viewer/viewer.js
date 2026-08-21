(function () {
  "use strict";

  const VIEWER_SCHEMA = "contraption.viewer/v3";
  const SCENE_SCHEMA = "contraption.physical-scene/v1";
  const RENDER_BUNDLE_SCHEMA = "contraption.render-bundle/v1";
  const TRIANGLE_SURFACE_SCHEMA = "contraption.triangle-surface/v1";
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

  function requireInteger(value, label, nonnegative) {
    requireCondition(Number.isInteger(value), `${label} must be an integer`);
    if (nonnegative) requireCondition(value >= 0, `${label} must be non-negative`);
    return value;
  }

  function requireUniqueStrings(value, label) {
    const items = requireArray(value, label, false);
    const seen = new Set();
    items.forEach((item, index) => {
      const text = requireString(item, `${label}[${index}]`);
      requireCondition(!seen.has(text), `${label} must not contain duplicate values`);
      seen.add(text);
    });
    return items;
  }

  function validateFabricationStandard(value, label) {
    const standard = requireObject(value, label);
    requireKeys(standard,
      ["family", "authority", "document", "designation", "role"],
      ["revision", "uri", "nominal_diameter_m", "pitch_m", "gauge_awg"], label);
    ["family", "authority", "document", "designation", "role"].forEach((field) =>
      requireString(standard[field], `${label}.${field}`));
    ["revision", "uri"].forEach((field) => {
      if (standard[field] !== undefined) requireString(standard[field], `${label}.${field}`);
    });
    ["nominal_diameter_m", "pitch_m"].forEach((field) => {
      if (standard[field] !== undefined) requireNumber(standard[field], `${label}.${field}`, true);
    });
    if (standard.gauge_awg !== undefined) {
      requireInteger(standard.gauge_awg, `${label}.gauge_awg`, true);
      requireCondition(standard.gauge_awg <= 40, `${label}.gauge_awg must not exceed 40`);
    }
  }

  function validateFabricationEvidence(value, label) {
    const evidence = requireObject(value, label);
    requireKeys(evidence, ["kind", "source"], ["locator", "sha256", "page"], label);
    requireString(evidence.kind, `${label}.kind`);
    requireString(evidence.source, `${label}.source`);
    if (evidence.locator !== undefined) requireString(evidence.locator, `${label}.locator`);
    if (evidence.sha256 !== undefined) {
      requireCondition(HASH_PATTERN.test(evidence.sha256), `${label}.sha256 is not canonical`);
    }
    if (evidence.page !== undefined) {
      requireInteger(evidence.page, `${label}.page`, false);
      requireCondition(evidence.page > 0, `${label}.page must be positive`);
    }
  }

  function validateRetention(value, label) {
    const retention = requireObject(value, label);
    requireKeys(retention, ["method"],
      ["hardware", "quantity", "torque_n_m", "locking_method", "installation_process"], label);
    requireString(retention.method, `${label}.method`);
    if (retention.hardware !== undefined) validateFabricationStandard(retention.hardware, `${label}.hardware`);
    if (retention.quantity !== undefined) {
      requireInteger(retention.quantity, `${label}.quantity`, false);
      requireCondition(retention.quantity > 0, `${label}.quantity must be positive`);
    }
    if (retention.torque_n_m !== undefined) {
      requireNumber(retention.torque_n_m, `${label}.torque_n_m`, false);
      requireCondition(retention.torque_n_m >= 0, `${label}.torque_n_m must be non-negative`);
    }
    ["locking_method", "installation_process"].forEach((field) => {
      if (retention[field] !== undefined) requireString(retention[field], `${label}.${field}`);
    });
  }

  function validateBearing(value, label) {
    const bearing = requireObject(value, label);
    requireKeys(bearing, ["method"], ["standard", "designation", "bore_diameter_m",
      "outer_diameter_m", "width_m", "radial_clearance_m", "axial_retention", "lubrication"], label);
    requireString(bearing.method, `${label}.method`);
    if (bearing.standard !== undefined) validateFabricationStandard(bearing.standard, `${label}.standard`);
    ["designation", "lubrication"].forEach((field) => {
      if (bearing[field] !== undefined) requireString(bearing[field], `${label}.${field}`);
    });
    ["bore_diameter_m", "outer_diameter_m", "width_m"].forEach((field) => {
      if (bearing[field] !== undefined) requireNumber(bearing[field], `${label}.${field}`, true);
    });
    if (bearing.radial_clearance_m !== undefined) {
      requireNumber(bearing.radial_clearance_m, `${label}.radial_clearance_m`, false);
      requireCondition(bearing.radial_clearance_m >= 0, `${label}.radial_clearance_m must be non-negative`);
    }
    if (bearing.axial_retention !== undefined) {
      validateRetention(bearing.axial_retention, `${label}.axial_retention`);
    }
  }

  function validateConductor(value, label) {
    const conductor = requireObject(value, label);
    requireKeys(conductor, [], ["standard", "conductor_count", "material", "cross_section_m2",
      "insulation_standard", "voltage_rating_v", "temperature_rating_k"], label);
    requireCondition(Object.keys(conductor).length > 0, `${label} must contain a known value`);
    ["standard", "insulation_standard"].forEach((field) => {
      if (conductor[field] !== undefined) {
        validateFabricationStandard(conductor[field], `${label}.${field}`);
      }
    });
    if (conductor.conductor_count !== undefined) {
      requireInteger(conductor.conductor_count, `${label}.conductor_count`, false);
      requireCondition(conductor.conductor_count > 0, `${label}.conductor_count must be positive`);
    }
    if (conductor.material !== undefined) requireString(conductor.material, `${label}.material`);
    ["cross_section_m2", "voltage_rating_v", "temperature_rating_k"].forEach((field) => {
      if (conductor[field] !== undefined) requireNumber(conductor[field], `${label}.${field}`, true);
    });
  }

  function validateTermination(value, label) {
    const termination = requireObject(value, label);
    requireKeys(termination, ["method", "installation_process"], ["hardware", "contact_part_number",
      "housing_part_number", "pin", "contact_pitch_m", "pad_dimensions_m"], label);
    requireString(termination.method, `${label}.method`);
    requireString(termination.installation_process, `${label}.installation_process`);
    if (termination.hardware !== undefined) {
      validateFabricationStandard(termination.hardware, `${label}.hardware`);
    }
    ["contact_part_number", "housing_part_number", "pin"].forEach((field) => {
      if (termination[field] !== undefined) requireString(termination[field], `${label}.${field}`);
    });
    if (termination.contact_pitch_m !== undefined) {
      requireNumber(termination.contact_pitch_m, `${label}.contact_pitch_m`, true);
    }
    if (termination.pad_dimensions_m !== undefined) {
      requireVector(termination.pad_dimensions_m, `${label}.pad_dimensions_m`, 2)
        .forEach((item, index) => requireCondition(item > 0,
          `${label}.pad_dimensions_m[${index}] must be positive`));
    }
  }

  function validateProtection(value, label) {
    const protection = requireObject(value, label);
    requireKeys(protection, ["kind"], ["standard", "part_number", "current_rating_a", "voltage_rating_v"], label);
    requireString(protection.kind, `${label}.kind`);
    if (protection.standard !== undefined) {
      validateFabricationStandard(protection.standard, `${label}.standard`);
    }
    if (protection.part_number !== undefined) requireString(protection.part_number, `${label}.part_number`);
    ["current_rating_a", "voltage_rating_v"].forEach((field) => {
      if (protection[field] !== undefined) requireNumber(protection[field], `${label}.${field}`, true);
    });
  }

  function validateRoute(value, label) {
    const route = requireObject(value, label);
    requireKeys(route, ["topology", "routed_length_m", "minimum_bend_radius_m",
      "service_loop_m", "strain_relief", "waypoints"], [], label);
    requireString(route.topology, `${label}.topology`);
    requireString(route.strain_relief, `${label}.strain_relief`);
    requireNumber(route.routed_length_m, `${label}.routed_length_m`, true);
    ["minimum_bend_radius_m", "service_loop_m"].forEach((field) => {
      requireNumber(route[field], `${label}.${field}`, false);
      requireCondition(route[field] >= 0, `${label}.${field} must be non-negative`);
    });
    requireUniqueStrings(route.waypoints, `${label}.waypoints`);
  }

  function validateTravel(value, label) {
    const travel = requireObject(value, label);
    requireKeys(travel, ["kind", "unit"], ["minimum", "maximum"], label);
    requireString(travel.kind, `${label}.kind`);
    requireString(travel.unit, `${label}.unit`);
    ["minimum", "maximum"].forEach((field) => {
      if (travel[field] !== undefined) requireNumber(travel[field], `${label}.${field}`, false);
    });
  }

  function validateFabricationRecord(value, label) {
    const fabrication = requireObject(value, label);
    requireKeys(fabrication, ["kind", "status", "missing"], ["standards", "retention",
      "bearing", "conductor", "termination", "protection", "route", "travel",
      "alignment_tolerance_m", "alignment_tolerance_rad", "evidence"], label);
    requireCondition(["fixed_mount", "rotary_support", "electrical_termination",
      "optical_alignment", "other"].includes(fabrication.kind), `${label}.kind is unsupported`);
    requireCondition(["missing", "partial", "specified"].includes(fabrication.status),
      `${label}.status is unsupported`);
    requireUniqueStrings(fabrication.missing, `${label}.missing`);
    if (fabrication.standards !== undefined) {
      requireArray(fabrication.standards, `${label}.standards`, false).forEach((item, index) =>
        validateFabricationStandard(item, `${label}.standards[${index}]`));
    }
    const nested = {
      retention: validateRetention,
      bearing: validateBearing,
      conductor: validateConductor,
      termination: validateTermination,
      protection: validateProtection,
      route: validateRoute,
      travel: validateTravel,
    };
    Object.entries(nested).forEach(([field, validator]) => {
      if (fabrication[field] !== undefined) validator(fabrication[field], `${label}.${field}`);
    });
    ["alignment_tolerance_m", "alignment_tolerance_rad"].forEach((field) => {
      if (fabrication[field] !== undefined) {
        requireNumber(fabrication[field], `${label}.${field}`, false);
        requireCondition(fabrication[field] >= 0, `${label}.${field} must be non-negative`);
      }
    });
    if (fabrication.evidence !== undefined) {
      requireArray(fabrication.evidence, `${label}.evidence`, false).forEach((item, index) =>
        validateFabricationEvidence(item, `${label}.evidence[${index}]`));
    }
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

  function matrixFromPose(pose) {
    const [tx, ty, tz] = pose.translation_m;
    const [w, x, y, z] = pose.rotation_quaternion_wxyz;
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), tx,
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), ty,
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), tz,
      0, 0, 0, 1,
    ];
  }

  function validateGeometry(geometry, label) {
    requireObject(geometry, label);
    const kind = requireString(geometry.kind, `${label}.kind`);
    const fields = ["kind", "dimensions_m", "shape_uri", "shape_sha256", "surface_id"];
    if (kind === "box" || kind === "sphere" || kind === "cylinder") {
      requireKeys(geometry, fields, [], label);
      requireVector(geometry.dimensions_m, `${label}.dimensions_m`, 3).forEach((item, index) => {
        requireCondition(item > 0, `${label}.dimensions_m[${index}] must be greater than zero`);
      });
      ["shape_uri", "shape_sha256", "surface_id"].forEach((field) => {
        requireCondition(geometry[field] === null, `${label}.${field} must be null for ${kind}`);
      });
    } else if (kind === "shape") {
      requireKeys(geometry, fields, [], label);
      requireVector(geometry.dimensions_m, `${label}.dimensions_m`, 3).forEach((item, index) => {
        requireCondition(item > 0, `${label}.dimensions_m[${index}] must be greater than zero`);
      });
      const uri = requireString(geometry.shape_uri, `${label}.shape_uri`);
      requireCondition(!uri.startsWith("/") && !uri.includes("\\") &&
        uri.split("/").every((part) => part.length > 0 && part !== "." && part !== ".."),
      `${label}.shape_uri must remain below its static.part directory`);
      requireCondition(HASH_PATTERN.test(geometry.shape_sha256),
        `${label}.shape_sha256 is not canonical`);
      requireString(geometry.surface_id, `${label}.surface_id`);
    } else {
      fail(`${label}.kind ${JSON.stringify(kind)} is unsupported`);
    }
  }

  function validateRenderBundle(bundle, assemblySha256, frames, solidGeometry, opticalConnectorKeys, assemblyId) {
    requireObject(bundle, "payload.render_bundle");
    requireKeys(bundle,
      ["schema", "sha256", "assembly_sha256", "surfaces", "solid_bindings", "sensors", "observations"],
      [], "payload.render_bundle");
    requireCondition(bundle.schema === RENDER_BUNDLE_SCHEMA,
      `payload.render_bundle.schema must be ${RENDER_BUNDLE_SCHEMA}`);
    requireCondition(HASH_PATTERN.test(bundle.sha256), "payload.render_bundle.sha256 is not canonical");
    requireCondition(bundle.assembly_sha256 === assemblySha256,
      "payload.render_bundle is bound to another assembly");

    const surfaces = requireObject(bundle.surfaces, "payload.render_bundle.surfaces");
    const materialized = new Map();
    Object.entries(surfaces).forEach(([digest, surface]) => {
      const label = `payload.render_bundle.surfaces[${JSON.stringify(digest)}]`;
      requireCondition(HASH_PATTERN.test(digest), `${label} key is not a canonical hash`);
      requireObject(surface, label);
      requireKeys(surface,
        ["schema", "sha256", "shape_manifest_sha256", "shape_artifact_sha256", "shape_id", "surface_id",
          "source_surface_sha256", "vertices_m", "triangles", "vertex_normals",
          "vertex_rgba_linear", "materials", "triangle_materials", "vertex_uncertainty_m"], [], label);
      requireCondition(surface.schema === TRIANGLE_SURFACE_SCHEMA,
        `${label}.schema must be ${TRIANGLE_SURFACE_SCHEMA}`);
      requireCondition(surface.sha256 === digest, `${label}.sha256 does not match its map key`);
      requireCondition(HASH_PATTERN.test(surface.shape_manifest_sha256),
        `${label}.shape_manifest_sha256 is not canonical`);
      requireCondition(HASH_PATTERN.test(surface.shape_artifact_sha256),
        `${label}.shape_artifact_sha256 is not canonical`);
      requireString(surface.shape_id, `${label}.shape_id`);
      requireString(surface.surface_id, `${label}.surface_id`);
      requireCondition(HASH_PATTERN.test(surface.source_surface_sha256),
        `${label}.source_surface_sha256 is not canonical`);
      const vertices = requireArray(surface.vertices_m, `${label}.vertices_m`, true);
      requireCondition(vertices.length >= 3, `${label}.vertices_m must contain at least three vertices`);
      vertices.forEach((vertex, index) => requireVector(vertex, `${label}.vertices_m[${index}]`, 3));
      const triangles = requireArray(surface.triangles, `${label}.triangles`, true);
      triangles.forEach((triangle, triangleIndex) => {
        const triangleLabel = `${label}.triangles[${triangleIndex}]`;
        requireCondition(requireArray(triangle, triangleLabel, false).length === 3,
          `${triangleLabel} must contain exactly three indices`);
        triangle.forEach((item, index) => {
          requireInteger(item, `${triangleLabel}[${index}]`, true);
          requireCondition(item < vertices.length, `${triangleLabel}[${index}] is out of range`);
        });
        requireCondition(new Set(triangle).size === 3, `${triangleLabel} repeats a vertex`);
      });
      if (surface.vertex_normals !== null) {
        const normals = requireArray(surface.vertex_normals, `${label}.vertex_normals`, false);
        requireCondition(normals.length === vertices.length,
          `${label}.vertex_normals must match vertices_m in length`);
        normals.forEach((normal, index) => requireVector(normal, `${label}.vertex_normals[${index}]`, 3));
      }
      if (surface.vertex_rgba_linear !== null) {
        const colors = requireArray(surface.vertex_rgba_linear, `${label}.vertex_rgba_linear`, false);
        requireCondition(colors.length === vertices.length,
          `${label}.vertex_rgba_linear must match vertices_m in length`);
        colors.forEach((color, index) => requireVector(color,
          `${label}.vertex_rgba_linear[${index}]`, 4).forEach((item) => requireCondition(
          item >= 0 && item <= 1, `${label}.vertex_rgba_linear values must be in [0, 1]`)));
      }
      const materials = requireArray(surface.materials, `${label}.materials`, true);
      const materialIds = new Set();
      materials.forEach((material, index) => {
        const materialLabel = `${label}.materials[${index}]`;
        requireObject(material, materialLabel);
        requireKeys(material, ["id", "base_color_linear_rgba", "optical_material_sha256"], [], materialLabel);
        const id = requireString(material.id, `${materialLabel}.id`);
        requireCondition(!materialIds.has(id), `${label}.materials repeats id ${JSON.stringify(id)}`);
        materialIds.add(id);
        requireVector(material.base_color_linear_rgba, `${materialLabel}.base_color_linear_rgba`, 4)
          .forEach((item) => requireCondition(item >= 0 && item <= 1,
            `${materialLabel}.base_color_linear_rgba values must be in [0, 1]`));
        if (material.optical_material_sha256 !== null) {
          requireCondition(HASH_PATTERN.test(material.optical_material_sha256),
            `${materialLabel}.optical_material_sha256 is not canonical`);
        }
      });
      const triangleMaterials = requireArray(surface.triangle_materials,
        `${label}.triangle_materials`, false);
      requireCondition(triangleMaterials.length === triangles.length,
        `${label}.triangle_materials must match triangles in length`);
      triangleMaterials.forEach((item, index) => {
        requireInteger(item, `${label}.triangle_materials[${index}]`, true);
        requireCondition(item < materials.length,
          `${label}.triangle_materials[${index}] references an unknown material`);
      });
      if (surface.vertex_uncertainty_m !== null) {
        const uncertainty = requireArray(surface.vertex_uncertainty_m,
          `${label}.vertex_uncertainty_m`, false);
        requireCondition(uncertainty.length === vertices.length,
          `${label}.vertex_uncertainty_m must match vertices_m in length`);
        uncertainty.forEach((item, index) => {
          requireNumber(item, `${label}.vertex_uncertainty_m[${index}]`, false);
          requireCondition(item >= 0, `${label}.vertex_uncertainty_m[${index}] must be non-negative`);
        });
      }
      materialized.set(digest, surface);
    });

    const usedSurfaces = new Set();
    const boundSolids = new Set();
    requireArray(bundle.solid_bindings, "payload.render_bundle.solid_bindings", true)
      .forEach((binding, index) => {
        const label = `payload.render_bundle.solid_bindings[${index}]`;
        requireObject(binding, label);
        requireKeys(binding, ["component", "body", "solid", "surface_sha256"], [], label);
        const solidKey = `${requireString(binding.component, `${label}.component`)}/` +
          `${requireString(binding.body, `${label}.body`)}/${requireString(binding.solid, `${label}.solid`)}`;
        requireCondition(solidGeometry.has(solidKey), `${label} references unknown solid ${solidKey}`);
        requireCondition(!boundSolids.has(solidKey), `${label} repeats solid ${solidKey}`);
        requireCondition(materialized.has(binding.surface_sha256),
          `${label}.surface_sha256 references an absent surface`);
        const geometry = solidGeometry.get(solidKey);
        const surface = materialized.get(binding.surface_sha256);
        requireCondition(geometry.kind === "shape", `${label} binds non-shape geometry`);
        requireCondition(surface.shape_manifest_sha256 === geometry.shape_sha256,
          `${label} is not bound to the solid's exact shape manifest`);
        requireCondition(surface.surface_id === geometry.surface_id,
          `${label} differs from the solid's authored surface_id`);
        boundSolids.add(solidKey);
        usedSurfaces.add(binding.surface_sha256);
      });
    const missingBindings = Array.from(solidGeometry.keys()).filter((key) => !boundSolids.has(key));
    requireCondition(missingBindings.length === 0,
      `payload.render_bundle does not bind every solid; missing=${missingBindings.join(", ")}`);

    const sensors = new Map();
    requireArray(bundle.sensors, "payload.render_bundle.sensors", false).forEach((sensor, index) => {
      const label = `payload.render_bundle.sensors[${index}]`;
      requireObject(sensor, label);
      requireKeys(sensor, ["id", "display_name", "connector", "projection", "descriptor_sha256"], [], label);
      const id = requireString(sensor.id, `${label}.id`);
      requireCondition(!sensors.has(id), `${label} repeats sensor ${JSON.stringify(id)}`);
      requireString(sensor.display_name, `${label}.display_name`);
      requireCondition(opticalConnectorKeys.has(sensor.connector),
        `${label}.connector is not a spatial optical connector`);
      requireCondition(HASH_PATTERN.test(sensor.descriptor_sha256),
        `${label}.descriptor_sha256 is not canonical`);
      const projection = requireObject(sensor.projection, `${label}.projection`);
      requireKeys(projection,
        ["kind", "resolution_px", "focal_length_px", "principal_point_px", "clipping_m"], [],
        `${label}.projection`);
      requireCondition(projection.kind === "pinhole", `${label}.projection.kind must be pinhole`);
      const resolution = requireArray(projection.resolution_px, `${label}.projection.resolution_px`, false);
      requireCondition(resolution.length === 2, `${label}.projection.resolution_px must contain two values`);
      resolution.forEach((item, itemIndex) => {
        requireInteger(item, `${label}.projection.resolution_px[${itemIndex}]`, true);
        requireCondition(item > 0, `${label}.projection.resolution_px values must be positive`);
      });
      requireVector(projection.focal_length_px, `${label}.projection.focal_length_px`, 2)
        .forEach((item) => requireCondition(item > 0, `${label}.projection focal lengths must be positive`));
      requireVector(projection.principal_point_px, `${label}.projection.principal_point_px`, 2);
      const clipping = requireVector(projection.clipping_m, `${label}.projection.clipping_m`, 2);
      requireCondition(clipping[0] > 0 && clipping[1] > clipping[0],
        `${label}.projection.clipping_m must be positive and increasing`);
      sensors.set(id, sensor);
    });

    const observationKeys = new Set();
    requireArray(bundle.observations, "payload.render_bundle.observations", false)
      .forEach((observation, index) => {
        const label = `payload.render_bundle.observations[${index}]`;
        requireObject(observation, label);
        requireKeys(observation,
          ["id", "artifact_sha256", "frame_index", "sensor", "sensor_descriptor_sha256",
            "optical_scene_sha256", "assembly_id", "assembly_sha256", "assembly_frame",
            "mount_connector", "mount_transform_sha256", "pose_world_from_sensor_row_major", "layers"],
          [], label);
        requireString(observation.id, `${label}.id`);
        requireCondition(HASH_PATTERN.test(observation.artifact_sha256),
          `${label}.artifact_sha256 is not canonical`);
        const frameIndex = requireInteger(observation.frame_index, `${label}.frame_index`, true);
        requireCondition(frameIndex < frames.length, `${label}.frame_index is outside the scene frame range`);
        requireCondition(sensors.has(observation.sensor), `${label}.sensor is unknown`);
        const sensor = sensors.get(observation.sensor);
        requireCondition(observation.sensor_descriptor_sha256 === sensor.descriptor_sha256,
          `${label}.sensor_descriptor_sha256 differs from its sensor declaration`);
        requireCondition(HASH_PATTERN.test(observation.optical_scene_sha256),
          `${label}.optical_scene_sha256 is not canonical`);
        requireCondition(observation.assembly_id === assemblyId,
          `${label}.assembly_id differs from the rendered assembly`);
        requireCondition(observation.assembly_sha256 === assemblySha256,
          `${label}.assembly_sha256 differs from the rendered assembly`);
        requireCondition(observation.assembly_frame === "world",
          `${label}.assembly_frame must be world`);
        requireCondition(observation.mount_connector === sensor.connector,
          `${label}.mount_connector differs from its sensor declaration`);
        requireCondition(HASH_PATTERN.test(observation.mount_transform_sha256),
          `${label}.mount_transform_sha256 is not canonical`);
        const poseMatrix = requireVector(observation.pose_world_from_sensor_row_major,
          `${label}.pose_world_from_sensor_row_major`, 16);
        const physicalMatrix = matrixFromPose(frames[frameIndex].connector_poses[sensor.connector]);
        requireCondition(poseMatrix.every((item, itemIndex) =>
          Math.abs(item - physicalMatrix[itemIndex]) <= 1e-9),
        `${label} pose differs from the exact physical connector pose at its frame`);
        const observationKey = `${frameIndex}/${observation.sensor}`;
        requireCondition(!observationKeys.has(observationKey), `${label} duplicates ${observationKey}`);
        observationKeys.add(observationKey);
        const layers = requireObject(observation.layers, `${label}.layers`);
        requireCondition(Object.keys(layers).length > 0, `${label}.layers must not be empty`);
        const allowedModes = new Set(["rgb", "depth", "segmentation", "uncertainty", "reconstruction"]);
        Object.entries(layers).forEach(([mode, layer]) => {
          const layerLabel = `${label}.layers.${mode}`;
          requireCondition(allowedModes.has(mode), `${layerLabel} has an unsupported mode`);
          requireObject(layer, layerLabel);
          if (layer.kind === "raster") {
            requireKeys(layer,
              ["kind", "sha256", "source_observation_sha256", "source_output_sha256",
                "source_output_media_type", "source_output_dtype", "source_output_shape",
                "display_transform", "display_range", "media_type", "width_px", "height_px", "data_base64"],
              [], layerLabel);
            requireCondition(HASH_PATTERN.test(layer.sha256), `${layerLabel}.sha256 is not canonical`);
            requireCondition(layer.source_observation_sha256 === observation.artifact_sha256,
              `${layerLabel}.source_observation_sha256 differs from its observation`);
            requireCondition(HASH_PATTERN.test(layer.source_output_sha256),
              `${layerLabel}.source_output_sha256 is not canonical`);
            requireCondition(layer.source_output_media_type === "application/vnd.numpy.npy",
              `${layerLabel}.source_output_media_type is unsupported`);
            const expectedDtype = mode === "segmentation" ? "int32" : "float32";
            requireCondition(layer.source_output_dtype === expectedDtype,
              `${layerLabel}.source_output_dtype must be ${expectedDtype}`);
            requireCondition(layer.media_type === "image/png", `${layerLabel}.media_type must be image/png`);
            requireInteger(layer.width_px, `${layerLabel}.width_px`, true);
            requireInteger(layer.height_px, `${layerLabel}.height_px`, true);
            requireCondition(layer.width_px > 0 && layer.height_px > 0,
              `${layerLabel} dimensions must be positive`);
            requireCondition(layer.width_px === sensor.projection.resolution_px[0] &&
              layer.height_px === sensor.projection.resolution_px[1],
            `${layerLabel} dimensions differ from the sensor resolution`);
            const expectedShape = mode === "rgb"
              ? [layer.height_px, layer.width_px, 3] : [layer.height_px, layer.width_px];
            requireCondition(JSON.stringify(layer.source_output_shape) === JSON.stringify(expectedShape),
              `${layerLabel}.source_output_shape differs from the sensor resolution`);
            const transforms = {
              rgb: "linear-rgb-clamped-to-srgb8",
              depth: "depth-near-white-far-black",
              segmentation: "stable-integer-label-colors",
              uncertainty: "uncertainty-log-blue-yellow-infinite-magenta",
            };
            requireCondition(layer.display_transform === transforms[mode],
              `${layerLabel}.display_transform is not canonical`);
            if (mode === "rgb" || mode === "segmentation") {
              requireCondition(layer.display_range === null,
                `${layerLabel}.display_range must be null`);
            } else {
              requireVector(layer.display_range, `${layerLabel}.display_range`, 2);
            }
            requireString(layer.data_base64, `${layerLabel}.data_base64`);
            try {
              requireCondition(btoa(atob(layer.data_base64)) === layer.data_base64,
                `${layerLabel}.data_base64 must be canonical padded base64`);
            } catch (_error) {
              fail(`${layerLabel}.data_base64 is invalid`);
            }
          } else if (layer.kind === "surface") {
            requireCondition(mode === "reconstruction",
              `${layerLabel} surface layers are valid only for reconstruction`);
            requireKeys(layer,
              ["kind", "source_observation_sha256", "surface_sha256", "world_pose"], [], layerLabel);
            requireCondition(HASH_PATTERN.test(layer.source_observation_sha256),
              `${layerLabel}.source_observation_sha256 is not canonical`);
            requireCondition(materialized.has(layer.surface_sha256),
              `${layerLabel}.surface_sha256 references an absent surface`);
            requirePose(layer.world_pose, `${layerLabel}.world_pose`);
            usedSurfaces.add(layer.surface_sha256);
          } else {
            fail(`${layerLabel}.kind must be raster or surface`);
          }
        });
      });
    const unusedSurfaces = Array.from(materialized.keys()).filter((key) => !usedSurfaces.has(key));
    requireCondition(unusedSurfaces.length === 0,
      `payload.render_bundle contains unreferenced surfaces: ${unusedSurfaces.join(", ")}`);
    return bundle;
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
    requireKeys(payload, ["schema", "title", "assembly_sha256", "scene"], ["live", "render_bundle"], "payload");
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
    const solidKeys = new Set();
    const solidGeometry = new Map();
    const connectorKeys = new Set();
    const spatialConnectorKeys = new Set();
    const opticalConnectorKeys = new Set();
    let solidCount = 0;
    components.forEach((component, componentIndex) => {
      const componentLabel = `scene.components[${componentIndex}]`;
      requireObject(component, componentLabel);
      requireKeys(component, ["id", "part", "model", "physical_role", "bodies", "connectors"], [], componentLabel);
      const instanceId = requireString(component.id, `${componentLabel}.id`);
      requireCondition(!instanceId.includes("/"), `${componentLabel}.id cannot contain '/'`);
      requireCondition(!instanceIds.has(instanceId), `duplicate component id ${JSON.stringify(instanceId)}`);
      instanceIds.add(instanceId);
      requireString(component.part, `${componentLabel}.part`);
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
          solidKeys.add(`${instanceId}/${bodyId}/${solidId}`);
          solidGeometry.set(`${instanceId}/${bodyId}/${solidId}`, solid.geometry);
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
          ["fabrication"], connectorLabel);
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
          if (connector.domain === "optical") opticalConnectorKeys.add(`${instanceId}.${connectorId}`);
        }
        const provenance = requireObject(connector.provenance, `${connectorLabel}.provenance`);
        requireKeys(provenance, ["kind", "source", "reference"], [], `${connectorLabel}.provenance`);
        requireString(provenance.kind, `${connectorLabel}.provenance.kind`);
        requireString(provenance.source, `${connectorLabel}.provenance.source`);
        if (provenance.reference !== null) {
          requireString(provenance.reference, `${connectorLabel}.provenance.reference`);
        }
        if (connector.fabrication !== undefined && connector.fabrication !== null) {
          validateFabricationRecord(connector.fabrication, `${connectorLabel}.fabrication`);
        }
      });
    });
    requireCondition(solidCount > 0, "scene contains no renderable solids");

    const connections = requireArray(scene.connections, "scene.connections", false);
    const connectionIds = new Set();
    connections.forEach((connection, connectionIndex) => {
      const label = `scene.connections[${connectionIndex}]`;
      requireObject(connection, label);
      requireKeys(connection, ["id", "kind", "domain", "endpoints", "metadata"],
        ["joint", "implementation"], label);
      const id = requireString(connection.id, `${label}.id`);
      requireCondition(!connectionIds.has(id), `duplicate connection id ${JSON.stringify(id)}`);
      connectionIds.add(id);
      requireCondition(["power", "signal", "attachment", "constraint"].includes(connection.kind),
        `${label}.kind is unsupported`);
      if (connection.domain !== null) requireString(connection.domain, `${label}.domain`);
      const connectionMetadata = requireObject(connection.metadata, `${label}.metadata`);
      requireCondition(Object.keys(connectionMetadata).length === 0,
        `${label}.metadata must be empty; physical semantics require typed fields`);
      if (connection.implementation !== undefined && connection.implementation !== null) {
        validateFabricationRecord(connection.implementation, `${label}.implementation`);
      }
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
    if (payload.render_bundle !== undefined) {
      validateRenderBundle(payload.render_bundle, payload.assembly_sha256, frames,
        solidGeometry, opticalConnectorKeys, scene.contraption_id);
    } else {
      const shapeSolids = components.flatMap((component) => component.bodies.flatMap((body) =>
        body.solids.filter((solid) => solid.geometry.kind === "shape")
          .map((solid) => `${component.id}/${body.id}/${solid.id}`)));
      requireCondition(shapeSolids.length === 0,
        `shape geometry requires exact render-bundle bindings; missing=${shapeSolids.join(", ")}`);
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

  function meshFor(geometry, surface) {
    if (surface !== undefined) {
      return {
        vertices: surface.vertices_m,
        faces: surface.triangles,
        materials: surface.materials,
        faceMaterials: surface.triangle_materials,
        vertexColors: surface.vertex_rgba_linear,
        vertexUncertainty: surface.vertex_uncertainty_m,
        surfaceSha256: surface.sha256,
      };
    }
    if (geometry.kind === "box") return boxMesh(geometry.dimensions_m);
    if (geometry.kind === "sphere") return sphereMesh(geometry.dimensions_m);
    if (geometry.kind === "cylinder") return cylinderMesh(geometry.dimensions_m);
    if (geometry.kind === "shape") {
      fail("shape geometry has no exact canonical surface binding; refusing a bounding-box substitute");
    }
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

  function inverseTransformPose(point, pose) {
    const relative = point.map((value, index) => value - pose.translation_m[index]);
    const [w, x, y, z] = pose.rotation_quaternion_wxyz;
    return rotateQuaternion(relative, [w, -x, -y, -z]);
  }

  function materialColor(material) {
    const values = material.base_color_linear_rgba.slice(0, 3)
      .map((item) => item <= 0.0031308 ? 12.92 * item : 1.055 * Math.pow(item, 1 / 2.4) - 0.055)
      .map((item) => Math.round(Math.max(0, Math.min(1, item)) * 255));
    return `rgb(${values[0]},${values[1]},${values[2]})`;
  }

  function vertexColor(mesh, face) {
    if (!mesh.vertexColors) return null;
    const linear = [0, 1, 2, 3].map((channel) =>
      face.reduce((sum, vertexIndex) => sum + mesh.vertexColors[vertexIndex][channel], 0) /
        face.length);
    const values = linear.slice(0, 3)
      .map((item) => item <= 0.0031308 ? 12.92 * item : 1.055 * Math.pow(item, 1 / 2.4) - 0.055)
      .map((item) => Math.round(Math.max(0, Math.min(1, item)) * 255));
    return { color: `rgb(${values[0]},${values[1]},${values[2]})`, alpha: linear[3] };
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
      ["schema", "assembly_sha256", "controllers", "inputs", "values"], [], "live control schema");
    requireCondition(schema.schema === "contraption.live-controls/v2",
      "live control schema has an unsupported schema identifier");
    requireCondition(schema.assembly_sha256 === payload.assembly_sha256,
      "live control schema assembly hash differs from the rendered assembly");
    const controllers = requireArray(schema.controllers, "live control schema.controllers", false);
    const controllerIds = [];
    controllers.forEach((controller, index) => {
      const label = `live control schema.controllers[${index}]`;
      requireObject(controller, label);
      requireKeys(controller, ["id", "program_id", "version", "sha256"], [], label);
      const id = requireString(controller.id, `${label}.id`);
      requireCondition(!controllerIds.includes(id), `duplicate live controller ${JSON.stringify(id)}`);
      controllerIds.push(id);
      requireString(controller.program_id, `${label}.program_id`);
      requireString(controller.version, `${label}.version`);
      requireCondition(HASH_PATTERN.test(controller.sha256), `${label}.sha256 is not canonical`);
    });
    const declarations = requireArray(schema.inputs, "live control schema.inputs", false);
    const values = requireObject(schema.values, "live control schema.values");
    const declaredNames = [];
    const inputs = new Map();

    const heading = document.createElement("p");
    heading.className = "panel-kicker live-control-heading";
    heading.textContent = controllerIds.length > 0
      ? `LIVE / ${controllerIds.join(" + ")}`
      : "LIVE / OPEN LOOP";
    container.append(heading);

    declarations.forEach((declaration, index) => {
      const label = `live control schema.inputs[${index}]`;
      requireObject(declaration, label);
      requireKeys(declaration,
        ["name", "type", "default", "minimum", "maximum", "unit", "description", "consumers"], [], label);
      const name = requireString(declaration.name, `${label}.name`);
      requireCondition(!declaredNames.includes(name), `duplicate live input ${JSON.stringify(name)}`);
      declaredNames.push(name);
      requireString(declaration.unit, `${label}.unit`);
      requireCondition(typeof declaration.description === "string", `${label}.description must be a string`);
      const consumers = requireArray(declaration.consumers, `${label}.consumers`, true);
      consumers.forEach((consumer, consumerIndex) => {
        requireString(consumer, `${label}.consumers[${consumerIndex}]`);
      });
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

    const renderBundle = payload.render_bundle;
    const surfaceByHash = renderBundle ? renderBundle.surfaces : {};
    const surfaceBinding = new Map(renderBundle ? renderBundle.solid_bindings.map((binding) => [
      `${binding.component}/${binding.body}/${binding.solid}`,
      surfaceByHash[binding.surface_sha256],
    ]) : []);
    const sensors = renderBundle ? renderBundle.sensors : [];
    const sensorById = new Map(sensors.map((sensor) => [sensor.id, sensor]));
    const observationByKey = new Map(renderBundle ? renderBundle.observations.map((observation) => [
      `${observation.frame_index}/${observation.sensor}`, observation,
    ]) : []);
    const rasterImages = new Map();

    const solids = [];
    const connectors = [];
    const connectorByKey = new Map();
    scene.components.forEach((component) => {
      component.bodies.forEach((body) => {
        body.solids.forEach((solid) => {
          const key = `${component.id}/${body.id}`;
          const surface = surfaceBinding.get(`${key}/${solid.id}`);
          solids.push({
            component,
            body,
            solid,
            bodyKey: key,
            mesh: meshFor(solid.geometry, surface),
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
    let viewpoint = "orbit";
    let viewMode = "scene";
    const viewpointSelect = byId("viewpoint");
    const viewModeSelect = byId("view-mode");
    const opticalSummary = byId("optical-view-summary");
    sensors.forEach((sensor) => {
      const option = document.createElement("option");
      option.value = sensor.id;
      option.textContent = sensor.display_name;
      viewpointSelect.append(option);
    });

    function currentObservation() {
      if (viewpoint === "orbit") return undefined;
      return observationByKey.get(`${frame}/${viewpoint}`);
    }

    function updateViewModes() {
      const selected = viewMode;
      viewModeSelect.replaceChildren();
      const spatial = document.createElement("option");
      spatial.value = "scene";
      spatial.textContent = "Spatial scene";
      viewModeSelect.append(spatial);
      if (viewpoint !== "orbit") {
        const modes = new Set();
        (renderBundle ? renderBundle.observations : [])
          .filter((observation) => observation.sensor === viewpoint)
          .forEach((observation) => Object.keys(observation.layers).forEach((mode) => modes.add(mode)));
        ["rgb", "depth", "segmentation", "uncertainty", "reconstruction"]
          .filter((mode) => modes.has(mode)).forEach((mode) => {
            const option = document.createElement("option");
            option.value = mode;
            option.textContent = mode[0].toUpperCase() + mode.slice(1);
            viewModeSelect.append(option);
          });
      }
      viewMode = Array.from(viewModeSelect.options).some((option) => option.value === selected)
        ? selected : "scene";
      viewModeSelect.value = viewMode;
      const sensor = sensorById.get(viewpoint);
      opticalSummary.textContent = sensor
        ? `${sensor.connector} · ${sensor.projection.resolution_px[0]}×${sensor.projection.resolution_px[1]} px · ${sensor.descriptor_sha256}`
        : sensors.length
          ? `${sensors.length} calibrated optical POV${sensors.length === 1 ? "" : "s"} available.`
          : "No calibrated optical POV is supplied.";
    }
    viewpointSelect.addEventListener("change", () => {
      viewpoint = viewpointSelect.value;
      viewMode = "scene";
      updateViewModes();
      tooltip.hidden = true;
      render();
    });
    viewModeSelect.addEventListener("change", () => {
      viewMode = viewModeSelect.value;
      tooltip.hidden = true;
      render();
    });
    updateViewModes();

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
        mesh: item.mesh,
      };
    }

    function reconstructionWorld(layer) {
      const surface = surfaceByHash[layer.surface_sha256];
      const mesh = meshFor({ kind: "shape" }, surface);
      const item = {
        component: { id: "reconstruction" },
        body: { id: "world" },
        solid: { id: "posterior-surface" },
        mesh,
        color: "#65a9ff",
        reconstruction: true,
      };
      return {
        item,
        vertices: mesh.vertices.map((point) => transformPose(point, layer.world_pose)),
        faces: mesh.faces,
        mesh,
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

    function drawRasterLayer(layer, width, height) {
      drawGrid(width, height, 1);
      let image = rasterImages.get(layer.sha256);
      if (!image) {
        image = new Image();
        image.decoding = "async";
        image.addEventListener("load", () => render(), { once: true });
        image.addEventListener("error", () => render(), { once: true });
        image.src = `data:${layer.media_type};base64,${layer.data_base64}`;
        rasterImages.set(layer.sha256, image);
      }
      if (image.complete && image.naturalWidth > 0) {
        const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
        const targetWidth = image.naturalWidth * scale;
        const targetHeight = image.naturalHeight * scale;
        const x = (width - targetWidth) / 2;
        const y = (height - targetHeight) / 2;
        context.imageSmoothingEnabled = false;
        context.drawImage(image, x, y, targetWidth, targetHeight);
        context.fillStyle = "rgba(7,16,20,0.82)";
        context.fillRect(10, height - 31, Math.min(width - 20, 530), 21);
        context.fillStyle = "#eaf1f4";
        context.font = "12px ui-monospace, monospace";
        context.textAlign = "left";
        context.fillText(`${viewMode.toUpperCase()} · ${layer.sha256}`, 18, height - 16);
      } else {
        context.fillStyle = "#8da1aa";
        context.font = "14px system-ui";
        context.textAlign = "center";
        context.fillText("Loading verified optical raster…", width / 2, height / 2);
      }
    }

    function drawUnavailableLayer(width, height, sensor) {
      drawGrid(width, height, 1);
      context.fillStyle = "#ffbd66";
      context.font = "15px system-ui";
      context.textAlign = "center";
      context.fillText(`No hash-bound ${viewMode} layer for ${sensor.display_name} at frame ${frame + 1}`,
        width / 2, height / 2);
    }

    function connectorColor(domain) {
      if (domain === "electrical") return "#ffbd66";
      if (domain === "signal") return "#65a9ff";
      if (domain === "mechanical" || domain === "rigid_mechanical") return "#55e5c5";
      return "#eaf1f4";
    }

    render = function renderScene() {
      const { width, height, ratio } = canvasSize();
      const selectedSensor = sensorById.get(viewpoint);
      const observation = currentObservation();
      const opticalLayer = viewMode === "scene" || !observation
        ? undefined : observation.layers[viewMode];
      if (viewMode !== "scene" && !opticalLayer) {
        drawUnavailableLayer(width, height, selectedSensor);
        hitRegions = [];
        connectorHitRegions = [];
        timeline.value = String(frame);
        byId("time-value").textContent = `${frames[frame].time_s.toFixed(3)} s`;
        byId("frame-value").textContent = `frame ${frame + 1} / ${frames.length}`;
        statusText.textContent = `Unavailable ${viewMode} observation · no fallback rendered`;
        return;
      }
      if (opticalLayer && opticalLayer.kind === "raster") {
        drawRasterLayer(opticalLayer, width, height);
        hitRegions = [];
        connectorHitRegions = [];
        timeline.value = String(frame);
        byId("time-value").textContent = `${frames[frame].time_s.toFixed(3)} s`;
        byId("frame-value").textContent = `frame ${frame + 1} / ${frames.length}`;
        statusText.textContent = `${viewMode.toUpperCase()} observation verified · ${opticalLayer.source_observation_sha256.slice(7, 19)}`;
        return;
      }
      drawGrid(width, height, ratio);
      const world = opticalLayer && opticalLayer.kind === "surface"
        ? [reconstructionWorld(opticalLayer)] : solids.map(solidWorld);
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
      const orbitProject = (point) => {
        const rotated = transformed(point);
        const relative = rotated.map((value, index) => value - center[index]);
        const perspective = 1 / Math.max(0.55, 1 + relative[2] / (extent * 5));
        return [
          width / 2 + view.panX * ratio + relative[0] * scale * perspective,
          height / 2 + view.panY * ratio - relative[1] * scale * perspective,
          relative[2],
          true,
        ];
      };
      let project = orbitProject;
      if (selectedSensor) {
        const sensorPose = frames[frame].connector_poses[selectedSensor.connector];
        const projection = selectedSensor.projection;
        const viewportScale = Math.min(
          width / projection.resolution_px[0], height / projection.resolution_px[1]
        );
        const offsetX = (width - projection.resolution_px[0] * viewportScale) / 2;
        const offsetY = (height - projection.resolution_px[1] * viewportScale) / 2;
        project = (point) => {
          const camera = inverseTransformPose(point, sensorPose);
          const depth = camera[2];
          const visible = depth >= projection.clipping_m[0] && depth <= projection.clipping_m[1];
          const safeDepth = Math.max(depth, projection.clipping_m[0]);
          const pixelX = projection.focal_length_px[0] * camera[0] / safeDepth +
            projection.principal_point_px[0];
          const pixelY = projection.principal_point_px[1] +
            projection.focal_length_px[1] * camera[1] / safeDepth;
          return [
            offsetX + pixelX * viewportScale,
            offsetY + pixelY * viewportScale,
            depth,
            visible,
          ];
        };
      }
      const faces = [];
      world.forEach((worldSolid) => {
        const projected = worldSolid.vertices.map(project);
        worldSolid.faces.forEach((face, faceIndex) => {
          const points = face.map((index) => projected[index]);
          if (selectedSensor && points.some((point) => !point[3])) return;
          const materialIndex = worldSolid.mesh.faceMaterials
            ? worldSolid.mesh.faceMaterials[faceIndex] : null;
          const material = materialIndex === null
            ? null : worldSolid.mesh.materials[materialIndex];
          const vertexAppearance = vertexColor(worldSolid.mesh, face);
          faces.push({
            item: worldSolid.item,
            points,
            depth: points.reduce((sum, point) => sum + point[2], 0) / points.length,
            shade: 0.72 + (faceIndex % 4) * 0.09,
            color: vertexAppearance
              ? vertexAppearance.color
              : material ? materialColor(material) : worldSolid.item.color,
            materialAlpha: vertexAppearance
              ? vertexAppearance.alpha
              : material ? material.base_color_linear_rgba[3] : 1,
          });
        });
      });
      faces.sort(selectedSensor
        ? (left, right) => right.depth - left.depth
        : (left, right) => left.depth - right.depth);
      hitRegions = [];
      faces.forEach((face) => {
        const alpha = Math.max(0.01, Math.min(1,
          globalAlpha * (alphaValues[face.item.component.id] === undefined
            ? 1 : alphaValues[face.item.component.id]) * face.materialAlpha));
        context.save();
        context.beginPath();
        context.moveTo(face.points[0][0], face.points[0][1]);
        face.points.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
        context.closePath();
        context.globalAlpha = alpha * 0.82;
        context.fillStyle = rgba(face.color, 1, face.shade);
        context.fill();
        context.globalAlpha = Math.min(0.7, alpha + 0.12);
        context.strokeStyle = "rgb(225,245,247)";
        context.lineWidth = Math.max(0.55, ratio * 0.65);
        context.stroke();
        context.restore();
        hitRegions.push({ item: face.item, points: face.points });
      });
      connectorHitRegions = [];
      if (showConnectors && viewpoint === "orbit") {
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
        sensors.forEach((sensor) => {
          const pose = frames[frame].connector_poses[sensor.connector];
          const projection = sensor.projection;
          const distance = extent * 0.18;
          const pixelCorners = [
            [0, 0], [projection.resolution_px[0], 0],
            [projection.resolution_px[0], projection.resolution_px[1]],
            [0, projection.resolution_px[1]],
          ];
          const worldCorners = pixelCorners.map(([pixelX, pixelY]) => transformPose([
            (pixelX - projection.principal_point_px[0]) * distance / projection.focal_length_px[0],
            (pixelY - projection.principal_point_px[1]) * distance / projection.focal_length_px[1],
            distance,
          ], pose));
          const origin = project(pose.translation_m);
          const projectedCorners = worldCorners.map(project);
          context.save();
          context.strokeStyle = "rgba(101,169,255,0.72)";
          context.lineWidth = Math.max(0.9, ratio);
          projectedCorners.forEach((corner) => {
            context.beginPath();
            context.moveTo(origin[0], origin[1]);
            context.lineTo(corner[0], corner[1]);
            context.stroke();
          });
          context.beginPath();
          context.moveTo(projectedCorners[0][0], projectedCorners[0][1]);
          projectedCorners.slice(1).forEach((corner) => context.lineTo(corner[0], corner[1]));
          context.closePath();
          context.stroke();
          context.restore();
        });
      }
      timeline.value = String(frame);
      byId("time-value").textContent = `${frames[frame].time_s.toFixed(3)} s`;
      byId("frame-value").textContent = `frame ${frame + 1} / ${frames.length}`;
      statusText.textContent = selectedSensor
        ? `${selectedSensor.display_name} POV · exact connector frame ${selectedSensor.connector}`
        : opticalLayer && opticalLayer.kind === "surface"
          ? `Reconstruction surface verified · ${opticalLayer.source_observation_sha256.slice(7, 19)}`
          : `Canonical assembly verified · ${scene.assembly_sha256.slice(7, 19)}`;
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
      if (viewpoint !== "orbit" || viewMode !== "scene") return;
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
      if (viewpoint !== "orbit" || viewMode !== "scene") return;
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

    byId("model-summary").textContent = `${scene.components.length} components \u00b7 ${solids.length} solids \u00b7 ${connectors.length} spatial connectors \u00b7 ${sensors.length} optical POVs \u00b7 ${frames.length} pose frames`;
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
