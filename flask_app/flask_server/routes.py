import os
import secrets
from PIL import Image
from flask_server import app, db, bcrypt
from flask import render_template, url_for, flash, redirect, request, abort
from flask_server.models import User, Post
from flask_server.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm
from flask_login import login_user, current_user,logout_user, login_required
from flask_server.transform_to_images import generate
from flask_server.keras_models import predict_image, load_image_from_paths
from pyrsgis import raster

@app.route("/")
@app.route("/home")
def home():
    if current_user.is_authenticated:
        #posts = Post.query.filter(Post.user_id == current_user.id).all()
        posts = Post.query.order_by(db.desc('id')).all()
        return render_template('home.html', posts=posts)
    else:
        return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user=User(username=form.username.data, email=form.email.data , password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! you are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user=User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else  redirect(url_for('home'))
        flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn

@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
           picture_file = save_picture(form.picture.data)
           current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET' :
        form.username.data == current_user.username
        form.email.data == current_user.email
    image_file= url_for('static',filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='account', image_file=image_file, form= form)

def save_picture_post(form_picture, name):
    random_hex = secrets.token_hex(8)
    picture_fn = random_hex + name
    picture_path = os.path.join(app.root_path, 'static/post_picture', picture_fn)
    form_picture.save(picture_path)
    return picture_fn

@app.route("/post/new",  methods=['GET', 'POST'])
@login_required
def new_post():
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    form = PostForm()
    if form.validate_on_submit():
        if form.picture.data :

            data_to_import = [form.picture.data, form.msi.data, form.cwi.data, form.lai.data]
            image_paths = []

            for f in data_to_import:
                if f:
                    random_hex = secrets.token_hex(8)
                    _, f_ext = os.path.splitext(f.filename)
                    filename = random_hex + f_ext
                    f.save(os.path.join(app.root_path, 'static', 'post_picture', filename))
                    image_paths.append(os.path.join('flask_server', 'static', 'post_picture', filename))
                else:
                    image_paths.append(None)

            dataSource, input = load_image_from_paths(image_paths)
            print("Starting calculating output................")
            output = predict_image(input, form.country.data)
            print("Finished Output")

            tiff_name = random_hex + ".tif"
            tiff_path =  os.path.join(app.root_path, 'static/post_picture', tiff_name)
            raster.export(output, dataSource, tiff_path, dtype='int', bands='all')

            mask, msi, rgb, infra, mask_msi, mask_rgb, msi_rgb, mask_infra, rgb_infra, msi_infra, mask_msi_infra, mask_rgb_infra, msi_rgb_infra, msi_rgb_mask, all, kpis = generate(input, output, form.color1.data, form.color2.data, form.color3.data)
            post = Post(title= form.title.data, tiff = tiff_name, content=form.content.data, mask=save_picture_post(mask, "mask.png"),
            msi=save_picture_post(msi, "msi.png"), rgb=save_picture_post(rgb, "rgb.png"), mask_msi=save_picture_post(mask_msi, "mask_msi.png"),
            infra=save_picture_post(infra, "infra.png"), mask_rgb=save_picture_post(mask_rgb, "mask_rgb.png"), msi_rgb=save_picture_post(msi_rgb, "msi_rgb.png"),
            mask_infra=save_picture_post(mask_infra, "mask_infra.png"), rgb_infra=save_picture_post(rgb_infra, "rgb_infra.png"),
            msi_infra=save_picture_post(msi_infra, "msi_infra.png"), mask_msi_infra=save_picture_post(mask_msi_infra, "mask_msi_infra.png"),
            mask_rgb_infra=save_picture_post(mask_rgb_infra, "mask_rgb_infra.png"), msi_rgb_infra=save_picture_post(msi_rgb_infra, "msi_rgb_infra.png"),
            msi_rgb_mask=save_picture_post(msi_rgb_mask, "msi_rgb_mask.png"), all_imgs =save_picture_post(all, "all.png"), kpis =  kpis, author= current_user)
            os.remove(os.path.join(app.root_path, 'static', 'post_picture', filename))

        else :
            post = Post(title= form.title.data, content=form.content.data, author= current_user)

        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))

    return render_template('create_post.html', title='New Post', form = form, legend = 'New Post')

@app.route("/post/<int:post_id>")
def post(post_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    return render_template('post.html', title=post.title, post = post)

@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'] )
@login_required
def update_post(post_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title= form.title.data
        post.content = form.content.data
        if form.picture.data :
            data_to_import = [form.picture.data, form.msi.data, form.cwi.data, form.lai.data]
            image_paths = []
            #import picture
            for f in data_to_import:
                if f:
                    random_hex = secrets.token_hex(8)
                    _, f_ext = os.path.splitext(f.filename)
                    filename = random_hex + f_ext
                    f.save(os.path.join(app.root_path, 'static', 'post_picture', filename))
                    image_paths.append(os.path.join('flask_server', 'static', 'post_picture', filename))
                else:
                    image_paths.append(None)

            dataSource, input = load_image_from_paths(image_paths)

            print("Starting calculating output................")
            output = predict_image(input, form.country.data)
            print("Finished Output")

            tiff_name = random_hex + ".tif"
            tiff_path =  os.path.join(app.root_path, 'static/post_picture', tiff_name)
            raster.export(output, dataSource, tiff_path, dtype='int', bands='all')

            mask, msi, rgb, infra, mask_msi, mask_rgb, msi_rgb, mask_infra, rgb_infra, msi_infra, mask_msi_infra, mask_rgb_infra, msi_rgb_infra, msi_rgb_mask, all, kpis = generate(input, output, form.color1, form.color2, form.color3)
            post.tiff = tiff_name
            post.mask=save_picture_post(mask, "mask.png")
            post.msi=save_picture_post(msi, "msi.png")
            post.rgb=save_picture_post(rgb, "rgb.png")
            post.mask_msi=save_picture_post(mask_msi, "mask_msi.png")
            post.infra=save_picture_post(infra, "infra.png")
            post.mask_rgb=save_picture_post(mask_rgb, "mask_rgb.png")
            post.msi_rgb=save_picture_post(msi_rgb, "msi_rgb.png")
            post.mask_infra=save_picture_post(mask_infra, "mask_infra.png")
            post.rgb_infra=save_picture_post(rgb_infra, "rgb_infra.png")
            post.msi_infra=save_picture_post(msi_infra, "msi_infra.png")
            post.mask_msi_infra=save_picture_post(mask_msi_infra, "mask_msi_infra.png")
            post.mask_rgb_infra=save_picture_post(mask_rgb_infra, "mask_rgb_infra.png")
            post.msi_rgb_infra=save_picture_post(msi_rgb_infra, "msi_rgb_infra.png"),
            post.msi_rgb_mask=save_picture_post(msi_rgb_mask, "msi_rgb_mask.png")
            post.all_imgs =save_picture_post(all, "all.png")
            post.kpis=kpis
            os.remove(os.path.join(app.root_path, 'static', 'post_picture', filename))

        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post', form = form, legend = 'Update Post')

@app.route("/delete/<int:post_id>")
def delete(post_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    return redirect(url_for('home'))

#auxiliary function
def get_imgs_with_id(post, img_id):
    path = url_for ('static', filename= 'post_picture/' + post.msi)
    mask_style, msi_style, rgb_style, infra_style = "", "", "", ""
    msi, mask,rgb = 14,15,12
    color_text = "text-primary"
    if img_id % 16 == 1:
        path = url_for ('static', filename= 'post_picture/' + post.mask)
        mask_style = color_text
        msi, rgb = 2, 4
    elif img_id % 16 == 2:
        msi_style = color_text
        mask, rgb = 1, 4
    elif img_id % 16 == 3:
        path = url_for ('static', filename= 'post_picture/' + post.mask_msi)
        mask_style, msi_style = color_text, color_text
        rgb = 4
    elif img_id % 16 == 4:
        path = url_for ('static', filename= 'post_picture/' + post.rgb)
        rgb_style = color_text
        mask, msi = 1, 2
    elif img_id % 16 == 5:
        path = url_for ('static', filename= 'post_picture/' + post.mask_rgb)
        mask_style, rgb_style = color_text, color_text
        msi = 2
    elif img_id % 16 == 6:
        path = url_for ('static', filename= 'post_picture/' + post.msi_rgb)
        msi_style, rgb_style = color_text, color_text
        mask = 1
    elif img_id % 16 == 7:
        path = url_for ('static', filename= 'post_picture/' + post.msi_rgb_mask)
        msi_style, rgb_style, mask_style = color_text, color_text, color_text
    if img_id % 16 == 8:
        path = url_for ('static', filename= 'post_picture/' + post.infra)
        infra_style = color_text
        msi, rgb, mask = 2,4,1
    elif img_id % 16 == 9:
        path = url_for ('static', filename= 'post_picture/' + post.mask_infra)
        mask_style, infra_style = color_text, color_text
        msi, rgb = 2, 4
    elif img_id % 16 == 10:
        path = url_for ('static', filename= 'post_picture/' + post.msi_infra)
        msi_style, infra_style = color_text, color_text
        mask, rgb = 1,4
    elif img_id % 16 == 11:
        path = url_for ('static', filename= 'post_picture/' + post.mask_msi_infra)
        mask_style, msi_style, infra_style = color_text, color_text, color_text
        rgb = 4
    elif img_id % 16 == 12:
        path = url_for ('static', filename= 'post_picture/' + post.rgb_infra)
        infra_style, rgb_style = color_text, color_text
        mask, msi = 1, 2
    elif img_id % 16 == 13:
        path = url_for ('static', filename= 'post_picture/' + post.mask_rgb_infra)
        infra_style, rgb_style, mask_style = color_text, color_text, color_text
        msi = 2
    elif img_id % 16 == 14:
        path = url_for ('static', filename= 'post_picture/' + post.msi_rgb_infra)
        msi_style, rgb_style, infra_style = color_text, color_text, color_text
        mask = 1
    elif img_id % 16 == 15:
        path = url_for ('static', filename= 'post_picture/' + post.all_imgs)
        msi_style, rgb_style, mask_style, infra_style = color_text, color_text, color_text, color_text
    return msi_style, mask_style, rgb_style, infra_style, mask, msi, rgb, path

@app.route("/viz/<int:post_id>/zoom/<int:zoom_size>/img/<int:zoom_id>")
def viz_zoom_img(post_id, zoom_size, zoom_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    zoom_msi_style, zoom_mask_style, zoom_rgb_style, zoom_infra_style, zoom_mask, zoom_msi, zoom_rgb, path_zoom = get_imgs_with_id(post, zoom_id)
    msi_style, mask_style, rgb_style, infra_style, mask, msi, rgb, path_img = get_imgs_with_id(post, img_id)
    return render_template('viz.html', title=post.title, kpis=post.kpis.split(";"),img_path = path_img, zoom_path = path_zoom, msi = msi, mask= mask, rgb= rgb,zoom_msi=zoom_msi, zoom_mask=zoom_mask, zoom_rgb= zoom_rgb, rgb_style=rgb_style, msi_style=msi_style, mask_style=mask_style,infra_style=infra_style,  zoom_msi_style = zoom_msi_style, zoom_mask_style=zoom_mask_style, zoom_rgb_style = zoom_rgb_style, zoom_infra_style = zoom_infra_style, img = img_id % 16, zoom_id = zoom_id %16 , zoom=zoom_size, post = post)

@app.route("/viz/<int:post_id>/zoom/<int:zoom_size>")
def viz_zoom(post_id, zoom_size):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    return render_template('viz.html', title=post.title, kpis=post.kpis.split(";"),img_path = url_for ('static', filename= 'post_picture/' + post.msi), zoom_path = url_for ('static', filename= 'post_picture/' + post.msi), msi = 6, mask= 1, zoom_mask = 1, zoom_msi = 6, rgb_style= "", msi_style="text-primary", mask_style="", infra_style="",  zoom_msi_style = "text-primary", zoom_mask_style="", zoom_rgb_style = "", zoom_infra_style = "", img = 2, zoom_id = 2, zoom=zoom_size, post = post)

@app.route("/viz/<int:post_id>/img/<int:img_id>/zoom/<int:zoom_size>/img/<int:zoom_id>")
def viz_img_zoom_img(post_id, zoom_size, img_id, zoom_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    zoom_msi_style, zoom_mask_style, zoom_rgb_style, zoom_infra_style, zoom_mask, zoom_msi, zoom_rgb, path_zoom = get_imgs_with_id(post, zoom_id)
    msi_style, mask_style, rgb_style, infra_style, mask, msi, rgb, path_img = get_imgs_with_id(post, img_id)
    return render_template('viz.html', title=post.title,kpis=post.kpis.split(";"), img_path = path_img, zoom_path = path_zoom, msi = msi, mask= mask, rgb= rgb, zoom_mask=zoom_mask, zoom_msi=zoom_msi, zoom_rgb= zoom_rgb, rgb_style=rgb_style, msi_style=msi_style, mask_style=mask_style, infra_style=infra_style,  zoom_msi_style = zoom_msi_style, zoom_mask_style=zoom_mask_style, zoom_rgb_style = zoom_rgb_style, zoom_infra_style = zoom_infra_style,  img = img_id % 16, zoom_id = zoom_id %16, zoom=zoom_size, post = post)

@app.route("/viz/<int:post_id>/img/<int:img_id>/zoom/<int:zoom_size>")
def viz_img_zoom(post_id, zoom_size, img_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    msi_style, mask_style, rgb_style, infra_style, mask, msi, rgb, path_img = get_imgs_with_id(post, img_id)
    return render_template('viz.html', title=post.title, kpis=post.kpis.split(";"),img_path = path_img, zoom_path = path_img, msi = msi, mask= mask, rgb= rgb, zoom_mask = mask, zoom_msi = msi, rgb_style=rgb_style, msi_style=msi_style, mask_style=mask_style, infra_style=infra_style,zoom_msi_style = msi_style, zoom_mask_style=mask_style, zoom_rgb_style = rgb_style, zoom_infra_style = infra_style, img = img_id % 16, zoom_id = img_id %16, zoom=zoom_size, post = post)

@app.route("/viz/<int:post_id>/img/<int:img_id>")
def viz_img(post_id, img_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    msi_style, mask_style, rgb_style, infra_style, mask, msi, rgb, path_img = get_imgs_with_id(post, img_id)
    return render_template('viz.html', title=post.title,kpis=post.kpis.split(";"), img_path = path_img, zoom_path = path_img, msi = msi, mask= mask, rgb= rgb, zoom_mask = mask, zoom_msi = msi, rgb_style= rgb_style, msi_style=msi_style, mask_style=mask_style, infra_style=infra_style,  zoom_msi_style = msi_style, zoom_mask_style=mask_style, zoom_rgb_style = rgb_style, zoom_infra_style = infra_style,  img = img_id % 16, zoom_id = img_id %16, zoom=3, post = post)

@app.route("/viz/<int:post_id>/img/<int:img_id>/img/<int:zoom_id>")
def viz_img_img(post_id, img_id, zoom_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    zoom_msi_style, zoom_mask_style, zoom_rgb_style, zoom_infra_style, zoom_mask, zoom_msi, zoom_rgb, path_zoom = get_imgs_with_id(zoom_id)
    msi_style, mask_style, rgb_style, infra_style, mask, msi, rgb, path_img = get_imgs_with_id(post, img_id)
    return render_template('viz.html', title=post.title,kpis=post.kpis.split(";"), img_path = path_img, zoom_path = path_zoom, msi = msi, mask= mask,  rgb= rgb, zoom_msi=zoom_msi, zoom_mask=zoom_mask, zoom_rgb=zoom_rgb, rgb_style= rgb_style, msi_style=msi_style, mask_style=mask_style, infra_style = infra_style,  zoom_msi_style = zoom_msi_style, zoom_mask_style=zoom_mask_style, zoom_rgb_style = zoom_rgb_style, zoom_infra_style = zoom_infra_style,  img = img_id % 16, zoom_id = zoom_id %16, zoom=3, post =post)

@app.route("/viz/<int:post_id>")
def viz(post_id):
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    post = Post.query.get_or_404(post_id)
    #if post.author != current_user:
    #    abort(403)
    return render_template('viz.html', title=post.title, kpis=post.kpis.split(";"), img_path = url_for ('static', filename= 'post_picture/' + post.msi), zoom_path = url_for ('static', filename= 'post_picture/' + post.msi), msi = 6, mask= 1, rgb=4, zoom_mask = 1, zoom_msi = 6, zoom_rgb = 4, rgb_style= "", msi_style="text-primary", mask_style="", infra_style = "", zoom_msi_style = "text-primary", zoom_mask_style="", zoom_rgb_style = "", zoom_infra_style = "", img = 2, zoom_id = 2, zoom=3, post = post)
