//! Shared mesh cook for triangle inputs in model-local space.
use crate::Error;
use glam::{Mat3, Mat4, Vec2, Vec3};
use sashiko_asset::{MeshBlob, MeshVertex};
use std::collections::HashMap;
type Result<T> = std::result::Result<T, Error>;

struct TangentGeometry<'a> {
    triangles: &'a [[u32; 3]],
    positions: &'a [[f32; 3]],
    normals: &'a [Vec3],
    uvs: &'a [[f32; 2]],
    tangents: Vec<[f32; 4]>,
}

impl bevy_mikktspace::Geometry for TangentGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.triangles.len()
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, corner: usize) -> [f32; 3] {
        self.positions[self.triangles[face][corner] as usize]
    }

    fn normal(&self, face: usize, corner: usize) -> [f32; 3] {
        self.normals[self.triangles[face][corner] as usize].to_array()
    }

    fn tex_coord(&self, face: usize, corner: usize) -> [f32; 2] {
        self.uvs[self.triangles[face][corner] as usize]
    }

    fn set_tangent(
        &mut self,
        tangent: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        corner: usize,
    ) {
        // A degenerate corner with no neighbor uses the reference MikkTSpace default.
        self.tangents[3 * face + corner] = tangent.unwrap_or_default().tangent_encoded();
    }
}

fn generate_tangents(
    indices: &[u32],
    positions: &[[f32; 3]],
    normals: &[Vec3],
    uvs: &[[f32; 2]],
) -> Result<Vec<[f32; 4]>> {
    let mut geometry = TangentGeometry {
        triangles: indices.as_chunks::<3>().0,
        positions,
        normals,
        uvs,
        tangents: vec![[0.; 4]; indices.len()],
    };
    bevy_mikktspace::generate_tangents(&mut geometry)?;
    Ok(geometry.tangents)
}

pub(crate) fn unit(v: Vec3, name: &str) -> Result<Vec3> {
    v.try_normalize()
        .ok_or_else(|| format!("invalid {name}").into())
}

pub(crate) enum Tangents {
    Authored(Vec<[f32; 4]>),
    GenerateGltf,
    GeneratePbrt,
}

pub(crate) fn triangles(
    positions: Vec<[f32; 3]>,
    normals: Vec<Vec3>,
    uvs: Vec<[f32; 2]>,
    tangents: Tangents,
    indices: Vec<u32>,
    transform: Mat4,
) -> Result<Vec<u8>> {
    if positions.len() != normals.len()
        || positions.len() != uvs.len()
        || matches!(&tangents, Tangents::Authored(t) if t.len() != positions.len())
        || positions.iter().any(|v| !Vec3::from(*v).is_finite())
        || uvs.iter().any(|v| !Vec2::from(*v).is_finite())
    {
        return Err("invalid or mismatched vertex attributes".into());
    }
    if indices.is_empty()
        || !indices.len().is_multiple_of(3)
        || indices.iter().any(|&i| i as usize >= positions.len())
    {
        return Err("incomplete or out-of-range triangle indices".into());
    }
    let generated = !matches!(tangents, Tangents::Authored(_));
    let tangent_data = match tangents {
        Tangents::Authored(data) => data,
        Tangents::GenerateGltf | Tangents::GeneratePbrt => {
            let mut data = generate_tangents(&indices, &positions, &normals, &uvs)?;
            if matches!(tangents, Tangents::GenerateGltf) {
                // With glTF's top-left UV origin, +Y in the normal map points
                // opposite the increasing-V bitangent returned by MikkTSpace.
                for tangent in &mut data {
                    tangent[3] = -tangent[3];
                }
            }
            data
        }
    };
    let linear = Mat3::from_mat4(transform);
    let det = linear.determinant();
    if !det.is_finite() || det.abs() < 1e-12 {
        return Err("singular node transform".into());
    }
    let normal_matrix = linear.inverse().transpose();
    let mut vertices = Vec::new();
    let mut remap = HashMap::new();
    let mut result = Vec::with_capacity(indices.len());
    for (corner, &index) in indices.iter().enumerate() {
        let i = index as usize;
        let [x, y, z, sign] = tangent_data[if generated { corner } else { i }];
        if !(sign == 1. || sign == -1.) || !Vec3::new(x, y, z).is_finite() {
            return Err("invalid tangent".into());
        }
        let n = normals[i];
        let t = Vec3::new(x, y, z);
        let t = match (t - n * n.dot(t)).try_normalize() {
            Some(value) => value,
            None if generated => n.any_orthonormal_vector(),
            None => return Err("authored tangent parallel to normal".into()),
        };
        let pos = transform.transform_point3(Vec3::from(positions[i]));
        if !pos.is_finite() {
            return Err("transformed position is not finite".into());
        }
        let normal = unit(normal_matrix * n, "transformed normal")?;
        let tangent = linear * t;
        let t = unit(
            tangent - normal * normal.dot(tangent),
            "transformed tangent",
        )?;
        let uv = uvs[i];
        let vertex = MeshVertex {
            position: pos.to_array(),
            normal: normal.to_array(),
            tangent: [t.x, t.y, t.z, sign * det.signum()],
            uv,
        };
        let key = (
            vertex.position.map(f32::to_bits),
            vertex.normal.map(f32::to_bits),
            vertex.tangent.map(f32::to_bits),
            vertex.uv.map(f32::to_bits),
        );
        let id = match remap.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let id = u32::try_from(vertices.len())?;
                vertices.push(vertex);
                *e.insert(id)
            }
        };
        result.push(id);
    }
    if det < 0. {
        for tri in result.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
    }
    Ok(MeshBlob {
        vertices,
        indices: result,
    }
    .encode()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_gltf_basis_matches_authored_on_a_normal_mapped_slope() -> Result<()> {
        // The encoded normal (0, 1, 2) represents the plane z = -y/2.
        for mirror_u in [false, true] {
            let (u0, u1, tx) = if mirror_u {
                (1., 0., -1.)
            } else {
                (0., 1., 1.)
            };
            for scale_x in [1., -1.] {
                for tangents in [
                    Tangents::GenerateGltf,
                    Tangents::Authored(vec![[tx, 0., 0., tx]; 3]),
                ] {
                    let mesh = MeshBlob::decode(&triangles(
                        vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                        vec![Vec3::Z; 3],
                        vec![[u0, 1.], [u1, 1.], [u0, 0.]],
                        tangents,
                        vec![0, 1, 2],
                        Mat4::from_scale(Vec3::new(scale_x, 1., 1.)),
                    )?)?;
                    for vertex in mesh.vertices {
                        let [x, y, z, w] = vertex.tangent;
                        let n = Vec3::from(vertex.normal);
                        let b = n.cross(Vec3::new(x, y, z)) * w;
                        assert_eq!(Vec3::new(x, y, z), Vec3::X * tx * scale_x);
                        assert_eq!(b + 2. * n, Vec3::new(0., 1., 2.));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mirrored_uv_faces_keep_separate_corner_tangents() -> Result<()> {
        let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]];
        let uvs = [[0., 0.], [1., 0.], [0., 1.], [0., 0.]];
        let tangents = generate_tangents(&[0, 1, 2, 1, 3, 2], &positions, &[Vec3::Z; 4], &uvs)?;
        assert_eq!(tangents[..3], [[1., 0., 0., 1.]; 3]);
        assert_eq!(tangents[3..], [[0., -1., 0., -1.]; 3]);
        Ok(())
    }

    #[test]
    fn isolated_degenerate_triangle_uses_reference_default() -> Result<()> {
        let tangents = generate_tangents(&[0, 0, 0], &[[0.; 3]], &[Vec3::Z], &[[0.; 2]])?;
        assert_eq!(tangents, [[1., 0., 0., -1.]; 3]);
        Ok(())
    }
}
