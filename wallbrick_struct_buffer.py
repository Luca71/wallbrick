from compushady import Buffer, HEAP_DEFAULT, HEAP_READBACK, HEAP_UPLOAD, Compute, config
import glfw
import compushady.formats
from compushady.shaders import hlsl
import struct
import platform
import random

compushady.config.set_debug(True)

print('Using device', compushady.get_current_device().name)

target = compushady.Texture2D(512, 512, compushady.formats.B8G8R8A8_UNORM)                       # creo texture

# we need space for 3 quads (uint4 * 2)
margin_offset = 15
paddle_size = [100, 10]


paddle =   [target.width // 2 - paddle_size[0] // 2,
            target.height - paddle_size[1] - margin_offset,
            paddle_size[0], 
            paddle_size[1],
            1, 1, 1, 1
            ]
ball = [target.width // 2, target.height // 4, 20, 20, 1, 1, 1, 1]
brick = [random.randint(100, target.width - 140),
        random.randint(40, 100),
        40, 25,
        1, 0, 0, 1] 

speed = 3 # the overall game speed

ball_direction = [1, 1] # initial ball direction

# to support d3d11 we are going to use two buffers here
quads_staging_buffer = compushady.Buffer(8 * 4 * 3, compushady.HEAP_UPLOAD)
quads_buffer = compushady.Buffer(
    quads_staging_buffer.size, format=compushady.formats.R32G32B32A32_SINT)

# our rendering system ;)
shader = hlsl.compile("""
struct data
{
    uint4 paddle;
    uint4 color;
};
StructuredBuffer<data> quads : register(t0);
RWTexture2D<float4> target : register(u0);

[numthreads(8, 8, 3)]
void main(int3 tid : SV_DispatchThreadID)
{
    data quad = quads[tid.z];
    if (tid.x > quad.paddle[0] + quad.paddle[2])
        return;
    if (tid.x < quad.paddle[0])
        return;
    if (tid.y < quad.paddle[1])
        return;
    if (tid.y > quad.paddle[1] + quad.paddle[3])
        return;

    target[tid.xy] = float4(quad.color);
}
""")

compute = compushady.Compute(shader, srv=[quads_buffer], uav=[target])

# a super simple clear screen procedure
clear_screen = compushady.Compute(hlsl.compile("""
RWTexture2D<float4> target : register(u0);

[numthreads(8, 8, 1)]
void main(int3 tid : SV_DispatchThreadID)
{
    target[tid.xy] = float4(0, 0, 0, 0);
}
"""), uav=[target])

glfw.init()
# we do not want implicit OpenGL!
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

window = glfw.create_window(target.width, target.height, 'Wall', None, None)

if platform.system() == 'Windows':
    swapchain = compushady.Swapchain(glfw.get_win32_window(
        window), compushady.formats.B8G8R8A8_UNORM, 2)
elif platform.system() == 'Darwin':
    from compushady.backends.metal import create_metal_layer
    ca_metal_layer = create_metal_layer(glfw.get_cocoa_window(window), compushady.formats.B8G8R8A8_UNORM)
    swapchain = compushady.Swapchain(
        ca_metal_layer, compushady.formats.B8G8R8A8_UNORM, 2)
else:
    swapchain = compushady.Swapchain((glfw.get_x11_display(), glfw.get_x11_window(
        window)), compushady.formats.B8G8R8A8_UNORM, 2)


def collide(source, dest):
    if source[0] + source[2] < dest[0]:
        return False
    if source[0] > dest[0] + dest[2]:
        return False
    if source[1] + source[3] < dest[1]:
        return False
    if source[1] > dest[1] + dest[3]:
        return False
    return True


while not glfw.window_should_close(window):
    glfw.poll_events()
    paddle_effect = None
    effect_variation = random.randrange(0, 2)
    if glfw.get_key(window, glfw.KEY_A):
        paddle[0] -= 1 * speed
        paddle_effect = -1 - effect_variation
    if glfw.get_key(window, glfw.KEY_D):
        paddle[0] += 1 * speed
        paddle_effect = 1 + effect_variation

    clear_screen.dispatch(target.width // 8, target.height // 8, 1)

    # ball vs paddle collisions
    if collide(ball, paddle):
        ball_direction[1] = -1
        if paddle_effect:
            ball_direction[0] = paddle_effect
    else:
        if ball[0] + ball[2] >= 512:
            ball_direction[0] = -1
        if ball[0] < 0:
            ball_direction[0] = 1
        if ball[1] + ball[3] >= 512:
            ball_direction[1] = -1
        if ball[1] < 0:
            ball_direction[1] = 1

    # ball vs brick collisions
    if collide(ball, brick):
        ball_direction[1] = -1
        if paddle_effect:
            ball_direction[0] = paddle_effect
        brick[0] = -50
        brick[1] = -20
    
    # paddle vs window collisions
    if paddle[0] < 0:
        paddle[0] = 0
    if paddle[0] + paddle[2] >= target.width:
        paddle[0] = target.width - paddle[2]

    ball[0] += ball_direction[0] * speed
    ball[1] += ball_direction[1] * speed

    quads_staging_buffer.upload(struct.pack('24i', *paddle, *ball, *brick))
    quads_staging_buffer.copy_to(quads_buffer)
    compute.dispatch(target.width // 8, target.height // 8, 1)
    swapchain.present(target)

swapchain = None  # this ensures the swapchain is destroyed before the window

glfw.terminate()