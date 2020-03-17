%
% Test code: Draw a grid of points
%

intrinsics = load('data/intrinsics.txt');
X = linspace(-2,2,10);
Y = linspace(-1,1,5);
uv = zeros([length(X)*length(Y), 2]);
k = 1;
for i=1:length(X)
    for j=1:length(Y)
        Z = 2;
        uv(k,:) = camera_to_fisheye(X(i), Y(j), Z, intrinsics);
        k = k + 1;
    end
end

scatter(uv(:,1), uv(:,2));
axis image;
grid on;
box on;
xlim([0, 1280]);
ylim([0, 720]);
set(gca, 'YDir', 'reverse');

function uv = camera_to_fisheye(X, Y, Z, intrinsics)
    % X,Y,Z:      3D point in camera coordinates. Z-axis pointing forward,
    %             Y-axis pointing down, and X-axis point to the right.
    % intrinsics: Fisheye intrinsics [f, cx, cy, k]

    f = intrinsics(1);
    cx = intrinsics(2);
    cy = intrinsics(3);
    k = intrinsics(4);
    theta = atan2(sqrt(X*X + Y*Y), Z);
    phi = atan2(Y, X);
    r = f*theta*(1 + k*theta*theta);
    u = cx + r*cos(phi);
    v = cy + r*sin(phi);
    uv = [u v];
end
