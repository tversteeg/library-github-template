use lyon::{
    math::Point,
    path::PathEvent,
    tessellation::{BuffersBuilder, FillAttributes, FillOptions, FillTessellator, VertexBuffers},
};
use miniquad::{fs, graphics::*, Context};
use std::{
    mem,
    sync::{Arc, Mutex},
};

type Vec2 = vek::Vec2<f64>;

const MAX_MESH_INSTANCES: usize = 1024 * 1024;

/// A reference to an uploaded vector path.
///
/// This contains an atomic reference counted mutex, which will unload the mesh from VRAM when
/// destructed.
#[derive(Debug, Clone)]
pub struct Mesh(Arc<Mutex<DrawCall>>);

impl Mesh {
    /// Render an instance of this mesh.
    ///
    /// Pretty slow because it needs to unlock the mutex. If possible use `draw_instances` instead.
    pub fn add_instance(&self, pos: Vec2) {
        let mut dc = self.0.lock().unwrap();

        dc.instances.push(Instance {
            position: [pos.x as f32, pos.y as f32],
        });
        assert!(dc.instances.len() < MAX_MESH_INSTANCES);

        // Tell the render loop that the data is out of date
        dc.refresh_instances = true;
    }

    /// Render a list of instances of this mesh.
    pub fn overwrite_instances(&self, pos: &Vec<Vec2>) {
        let mut dc = self.0.lock().unwrap();

        dc.instances = pos
            .iter()
            .map(|pos| Instance {
                position: [pos.x as f32, pos.y as f32],
            })
            .collect();
        assert!(dc.instances.len() < MAX_MESH_INSTANCES);

        // Tell the render loop that the data is out of date
        dc.refresh_instances = true;
    }

    /// Remove all instances.
    pub fn clear_instances(&self) {
        let mut dc = self.0.lock().unwrap();

        dc.instances.clear();
        dc.refresh_instances = true;
    }
}

/// A wrapper around the OpenGL calls so the main file won't be polluted.
pub struct Render {
    pipeline: Pipeline,
    /// A list of draw calls with bindings that will be generated.
    ///
    /// The draw calls are wrapped in a `Arc<Mutex<_>>` construction so it can be passed safely as
    /// a reference.
    draw_calls: Vec<Arc<Mutex<DrawCall>>>,
    /// Whether some draw calls are missing bindings.
    missing_bindings: bool,
}

impl Render {
    /// Setup the OpenGL pipeline and the texture for the framebuffer.
    pub fn new(ctx: &mut Context) -> Self {
        // Create an OpenGL pipeline
        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::META);
        let pipeline = Pipeline::new(
            ctx,
            &[
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("pos", VertexFormat::Float2, 0),
                VertexAttribute::with_buffer("inst_pos", VertexFormat::Float2, 1),
            ],
            shader,
        );

        Self {
            pipeline,
            draw_calls: vec![],
            missing_bindings: false,
        }
    }

    /// Upload a lyon path.
    ///
    /// Returns a reference that can be used to add instances.
    pub fn upload_path<P>(&mut self, path: P) -> Mesh
    where
        P: IntoIterator<Item = PathEvent>,
    {
        // Tessalate the path, converting it to vertices & indices
        let mut geometry: VertexBuffers<Vertex, u16> = VertexBuffers::new();
        let mut tessellator = FillTessellator::new();
        {
            tessellator
                .tessellate(
                    path,
                    &FillOptions::default(),
                    &mut BuffersBuilder::new(&mut geometry, |pos: Point, _: FillAttributes| {
                        Vertex {
                            pos: pos.to_array(),
                            ..Default::default()
                        }
                    }),
                )
                .unwrap();
        }
        let vertices = geometry.vertices.clone();
        let indices = geometry.indices.clone();

        // Create an OpenGL draw call for the path
        let draw_call = Arc::new(Mutex::new(DrawCall {
            vertices,
            indices,
            bindings: None,
            instances: vec![],
            instance_positions: vec![],
            refresh_instances: false,
        }));
        self.draw_calls.push(draw_call.clone());

        // Tell the next render loop to create bindings for this
        self.missing_bindings = true;

        // Return the draw call in a newtype struct so it can be used as a reference
        Mesh(draw_call)
    }

    /// Render the graphics.
    pub fn render(&mut self, ctx: &mut Context) {
        let (width, height) = ctx.screen_size();

        // Create bindings & update the instance vertices if necessary
        self.draw_calls.iter().for_each(|dc| {
            let mut dc = dc.lock().unwrap();

            // Create bindings if missing
            if self.missing_bindings && dc.bindings.is_none() {
                dc.create_bindings(ctx);
            }

            if dc.refresh_instances {
                // Upload the instance positions
                let bindings = dc.bindings.as_ref().unwrap();
                bindings.vertex_buffers[1].update(ctx, &dc.instances);

                dc.refresh_instances = false;
            }
        });

        self.missing_bindings = false;

        // Start rendering
        ctx.begin_default_pass(PassAction::Nothing);

        // Render the separate draw calls
        for dc in self.draw_calls.iter_mut() {
            let mut dc = dc.lock().unwrap();

            // Only render when we actually have instances
            if dc.instances.is_empty() {
                continue;
            }

            let bindings = dc.bindings.as_ref().unwrap();

            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_scissor_rect(0, 0, width as i32, height as i32);
            ctx.apply_bindings(bindings);
            ctx.apply_uniforms(&Uniforms {
                resolution: (width, height),
            });
            ctx.draw(0, dc.indices.len() as i32, dc.instances.len() as i32);
        }

        ctx.end_render_pass();

        ctx.commit_frame();
    }
}

/// A single uploaded mesh as a draw call.
#[derive(Debug)]
struct DrawCall {
    /// Render vertices, build by lyon path.
    vertices: Vec<Vertex>,
    /// Render indices, build by lyon path.
    indices: Vec<u16>,
    /// Position data for the instances.
    instance_positions: Vec<[f32; 2]>,
    /// Render bindings, generated on render loop if empty.
    bindings: Option<Bindings>,
    /// List of instances to render.
    instances: Vec<Instance>,
    /// Whether the instance information should be reuploaded to the GPU.
    refresh_instances: bool,
}

impl DrawCall {
    /// Create bindings if they are missing.
    fn create_bindings(&mut self, ctx: &mut Context) {
        // The vertex buffer of the vector paths
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &self.vertices);
        // The index buffer of the vector paths
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &self.indices);

        // A dynamic buffer that will contain all positions for all instances
        let instance_positions = Buffer::stream(
            ctx,
            BufferType::VertexBuffer,
            MAX_MESH_INSTANCES * mem::size_of::<Instance>(),
        );

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer, instance_positions],
            index_buffer,
            images: vec![],
        };
        self.bindings = Some(bindings);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
struct Vertex {
    pos: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Primitive {
    color: [f32; 4],
    translate: [f32; 2],
    z_index: i32,
    width: f32,
}

#[repr(C)]
#[derive(Debug)]
struct Uniforms {
    resolution: (f32, f32),
}

#[repr(C)]
#[derive(Debug)]
struct Instance {
    position: [f32; 2],
}

mod shader {
    use miniquad::graphics::*;

    pub const VERTEX: &str = r#"#version 100
attribute vec2 pos;
attribute vec2 inst_pos;

uniform vec2 resolution;

void main() {
    vec2 world_pos = (pos + inst_pos) / (vec2(0.5, -0.5) * resolution);

    gl_Position = vec4(world_pos, 0.0, 1.0);
}
"#;

    pub const FRAGMENT: &str = r#"#version 100

void main() {
    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}"#;

    pub const META: ShaderMeta = ShaderMeta {
        images: &[],
        uniforms: UniformBlockLayout {
            uniforms: &[("resolution", UniformType::Float2)],
        },
    };
}
